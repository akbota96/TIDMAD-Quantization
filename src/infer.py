"""FP32 Keras WaveNet inference on TIDMAD validation data.

Outputs preserve the source HDF5 structure and metadata while replacing
channel0001/timeseries with denoised signed int8 samples.

Use --benchmark-windows to test throughput without copying/writing HDF5.
Only completed files are published under the final output filename.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from model import build_wavenet

# Keep this BELOW the workstation's 16 GiB per-user VRAM quota.
# Configure before building the model or initializing the GPU.
GPUS = tf.config.list_physical_devices("GPU")
if GPUS:
    tf.config.set_logical_device_configuration(
        GPUS[0], [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
    )

SEQ_LEN = 40_000


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start-file", type=int, default=15)
    p.add_argument("--end-file", type=int, default=19)
    p.add_argument("--batch-size", type=int, default=4, help="Number of 40,000-sample windows per inference call")
    p.add_argument("--target-convention", choices=("shifted", "signed"), default="shifted")
    p.add_argument("--dry-run", action="store_true", help="Predict the first two windows, without writing outputs")
    p.add_argument("--benchmark-windows", type=int, default=0,
                   help="Process at most N windows per file without creating output; 0 means full inference")
    p.add_argument("--progress-every", type=int, default=100,
                   help="Report throughput after approximately N windows")
    args = p.parse_args()
    if args.batch_size < 1 or args.progress_every < 1 or args.benchmark_windows < 0:
        p.error("batch-size and progress-every must be positive; benchmark-windows must be nonnegative")
    if args.start_file > args.end_file:
        p.error("start-file must not exceed end-file")
    if args.dry_run and args.benchmark_windows:
        p.error("--dry-run and --benchmark-windows cannot be combined")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not args.data_dir.is_dir():
        raise NotADirectoryError(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_wavenet(seq_len=SEQ_LEN)
    model.load_weights(str(args.weights))
    print(f"Loaded FP32 weights: {args.weights}; parameters={model.count_params()}", flush=True)
    print(f"Target convention: {args.target_convention}; batch size={args.batch_size}", flush=True)

    # TensorFlow traces only once for variable batch dimension. No Keras predict()
    # per-batch Python overhead. XLA is optional and deliberately disabled here.
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, SEQ_LEN), dtype=tf.int32)], jit_compile=False)
    def predict_batch(x):
        return model(x, training=False)

    offset = 128 if args.target_convention == "shifted" else 0

    for file_num in range(args.start_file, args.end_file + 1):
        source = args.data_dir / f"abra_validation_{file_num:04d}.h5"
        dest = args.out_dir / f"abra_validation_denoised_wavenet_keras_mse_{file_num:04d}.h5"
        partial = args.out_dir / f"abra_validation_denoised_wavenet_keras_mse_{file_num:04d}.h5.partial"
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == dest.resolve():
            raise ValueError("Output must differ from input")
        if not (args.dry_run or args.benchmark_windows) and (dest.exists() or partial.exists()):
            raise FileExistsError(f"Output or partial file already exists: {dest} / {partial}")

        with h5py.File(source, "r") as src:
            squid = src["timeseries/channel0001/timeseries"]
            sg = src["timeseries/channel0002/timeseries"]
            total = squid.shape[0]
            if squid.ndim != 1 or sg.ndim != 1 or sg.shape[0] != total:
                raise ValueError(f"Unexpected channel shapes in {source}")
            if total % SEQ_LEN:
                raise ValueError(f"Number of samples is not divisible by {SEQ_LEN}: {source}")
            if squid.dtype != np.dtype("int8"):
                raise ValueError(f"Expected signed int8 input, got {squid.dtype}")
            n_windows = total // SEQ_LEN
            print(f"{source.name}: {total:,} samples, {n_windows:,} windows", flush=True)

            if args.dry_run:
                raw = np.asarray(squid[:2 * SEQ_LEN], dtype=np.int16)
                inp = (raw + 128).reshape(2, SEQ_LEN).astype(np.int32)
                pred = predict_batch(tf.convert_to_tensor(inp)).numpy().reshape(-1) - offset
                print(f"  DRY RUN: input std={raw.std():.4f}, prediction std={pred.std():.4f}, "
                      f"min={pred.min():.3f}, max={pred.max():.3f}; no output written", flush=True)
                continue

        benchmarking = args.benchmark_windows > 0
        requested_windows = min(n_windows, args.benchmark_windows) if benchmarking else n_windows
        if benchmarking:
            print(f"  BENCHMARK: {requested_windows:,} windows; no output will be written", flush=True)
        else:
            print(f"Copying original HDF5 (metadata preserved) -> {partial}", flush=True)
            # A partial filename avoids making an incomplete result look complete.
            shutil.copyfile(source, partial)

        run_start = time.perf_counter()
        processed = 0
        last_report = 0
        finite_min = float("inf")
        finite_max = float("-inf")
        try:
            # In benchmark mode, read the original source. In normal mode,
            # read/write the copied channel in place, preserving all other data.
            input_path = source if benchmarking else partial
            with h5py.File(input_path, "r" if benchmarking else "r+") as h5:
                out_ds = h5["timeseries/channel0001/timeseries"]
                for start_window in range(0, requested_windows, args.batch_size):
                    count = min(args.batch_size, requested_windows - start_window)
                    lo = start_window * SEQ_LEN
                    hi = lo + count * SEQ_LEN
                    raw = np.asarray(out_ds[lo:hi], dtype=np.int16)
                    inp = (raw + 128).reshape(count, SEQ_LEN).astype(np.int32)
                    pred = predict_batch(tf.convert_to_tensor(inp)).numpy()
                    if pred.shape != (count, SEQ_LEN, 1):
                        raise RuntimeError(f"Unexpected model output shape: {pred.shape}")
                    signed = pred.reshape(-1).astype(np.float32) - offset
                    if not np.isfinite(signed).all():
                        raise FloatingPointError(f"Nonfinite prediction at file {file_num}, window {start_window}")
                    finite_min = min(finite_min, float(signed.min()))
                    finite_max = max(finite_max, float(signed.max()))
                    if not benchmarking:
                        quantized = np.clip(np.rint(signed), -128, 127).astype(np.int8)
                        out_ds[lo:hi] = quantized
                    processed += count
                    if processed == requested_windows or processed - last_report >= args.progress_every:
                        elapsed = max(time.perf_counter() - run_start, 1e-9)
                        speed = processed / elapsed
                        eta = (requested_windows - processed) / speed if speed > 0 else float("inf")
                        print(f"  processed {processed:,}/{requested_windows:,} windows | "
                              f"{speed:.2f} windows/s | elapsed={elapsed:.1f}s | ETA={eta/60:.1f}min",
                              flush=True)
                        last_report = processed
                if not benchmarking:
                    h5.attrs["denoising_model"] = "WaveNet_MSE_FP32_Keras"
                    h5.attrs["model_output_target_convention"] = args.target_convention
                    h5.attrs["inference_weights_filename"] = args.weights.name
                    h5.flush()
            if not benchmarking:
                os.replace(partial, dest)
                print(f"COMPLETE: {dest}", flush=True)
            else:
                print(f"BENCHMARK COMPLETE: {processed:,} windows; "
                      f"prediction min={finite_min:.3f}, max={finite_max:.3f}", flush=True)
        except BaseException:
            if not benchmarking:
                print(f"INCOMPLETE OUTPUT LEFT AT: {partial} (not valid for scoring)", flush=True)
            raise


if __name__ == "__main__":
    main()
