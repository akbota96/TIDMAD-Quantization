"""Train or continue training a Keras FP32 MSE WaveNet on TIDMAD.

Reads HDF5 windows on demand. For each file in each epoch, samples one
random slot among 10 per group, permutes groups, and trains one window per
group. Preserves the original per-file Adam optimizer reset.
"""

import argparse
import csv
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from model import build_wavenet

# Configure the GPU before TensorFlow initializes it. ligroup enforces a
# 16 GiB per-user VRAM quota; the previously tested 10 GiB cap works.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
    )

SEQ_LEN = 40_000
SAMPLE_SIZE = 10
N_SAMPLES = 2_000_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="fresh_fp32_results")
    parser.add_argument("--start-file", type=int, default=15)
    parser.add_argument("--end-file", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of full passes over the specified training files")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Existing .weights.h5 checkpoint to load before training")
    parser.add_argument("--max-windows-per-file", type=int, default=0,
                        help="0 = all available; e.g. 20 for a smoke test")
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save intermediate weights every N updates per file; 0 disables")
    args = parser.parse_args()

    if args.end_file < args.start_file:
        parser.error("--end-file must be >= --start-file")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.max_windows_per_file < 0:
        parser.error("--max-windows-per-file must be nonnegative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"Checkpoint not found: {args.resume_from}")

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_wavenet()
    model.summary()
    print("Trainable parameters:", model.count_params(), flush=True)

    if args.resume_from is not None:
        model.load_weights(str(args.resume_from))
        print(f"Loaded initial weights from: {args.resume_from}", flush=True)
        print("NOTE: Optimizer state is not restored; Adam resets per file.", flush=True)

    csv_path = out / "training_log.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "file", "step", "mse"])
        fh.flush()

        for epoch in range(1, args.epochs + 1):
            print(f"=== EPOCH {epoch}/{args.epochs} ===", flush=True)

            for file_num in range(args.start_file, args.end_file + 1):
                fname = Path(args.data_dir) / f"abra_training_{file_num:04d}.h5"
                if not fname.exists():
                    raise FileNotFoundError(fname)

                # Match the original optimizer reset on each training file.
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                    loss="mse",
                )

                with h5py.File(fname, "r") as h5:
                    in_ds = h5["timeseries/channel0001/timeseries"]
                    tar_ds = h5["timeseries/channel0002/timeseries"]
                    available = min(len(in_ds), len(tar_ds), N_SAMPLES)
                    num_groups = available // (SEQ_LEN * SAMPLE_SIZE)
                    if num_groups < 1:
                        raise RuntimeError(f"Not enough samples in {fname}")

                    # Epoch 1 matches the original sampling seed. Subsequent
                    # epochs select new slots/permutations rather than replaying
                    # the same 25,000 training windows four times.
                    rng = np.random.default_rng(args.seed + file_num +
                                                (epoch - 1) * 100_000)
                    slot = int(rng.integers(0, SAMPLE_SIZE))
                    indices = rng.permutation(num_groups)
                    if args.max_windows_per_file:
                        indices = indices[:args.max_windows_per_file]

                    print(f"EPOCH {epoch} FILE {fname} "
                          f"slot={slot} batches={len(indices)}", flush=True)

                    for step, group in enumerate(indices, start=1):
                        start = (int(group) * SAMPLE_SIZE + slot) * SEQ_LEN
                        x = (np.asarray(in_ds[start:start + SEQ_LEN],
                                        dtype=np.int16) + 128)
                        y = (np.asarray(tar_ds[start:start + SEQ_LEN],
                                        dtype=np.float32) + 128.0)

                        if x.shape[0] != SEQ_LEN or y.shape[0] != SEQ_LEN:
                            raise RuntimeError(
                                f"Short window in {fname} at sample {start}")
                        if x.min() < 0 or x.max() > 255:
                            raise ValueError(
                                f"ADC embedding input outside [0, 255] in {fname}")

                        loss = float(model.train_on_batch(
                            x.astype("int32")[None, :],
                            y[None, :, None],
                            return_dict=False,
                        ))
                        writer.writerow([epoch, file_num, step, loss])

                        if args.checkpoint_every and step % args.checkpoint_every == 0:
                            ckpt = out / (
                                f"WaveNet_MSE_FP32_epoch{epoch:02d}"
                                f"_file{file_num:04d}_step{step:05d}.weights.h5"
                            )
                            model.save_weights(ckpt)
                            fh.flush()
                            print(f"Checkpoint saved: {ckpt}", flush=True)

                        if step == 1 or step % 100 == 0 or step == len(indices):
                            print(f" epoch={epoch} file={file_num} "
                                  f"step={step}/{len(indices)} "
                                  f"mse={loss:.6g}", flush=True)
                            fh.flush()

                ckpt = out / (
                    f"WaveNet_MSE_FP32_epoch{epoch:02d}"
                    f"_after_{file_num:04d}.weights.h5"
                )
                model.save_weights(ckpt)
                print(f"Saved: {ckpt}", flush=True)

            epoch_ckpt = out / f"WaveNet_MSE_FP32_epoch{epoch:02d}.weights.h5"
            model.save_weights(epoch_ckpt)
            print(f"EPOCH {epoch} COMPLETE: {epoch_ckpt}", flush=True)
            fh.flush()

    final_ckpt = out / "WaveNet_MSE_FP32_final.weights.h5"
    model.save_weights(final_ckpt)
    print(f"Saved: {final_ckpt}", flush=True)


if __name__ == "__main__":
    main()
