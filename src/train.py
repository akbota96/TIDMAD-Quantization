"""Train a fresh Keras FP32 MSE WaveNet on TIDMAD HDF5 training files.

Reads input/target windows on demand to avoid loading full 2-billion-sample
channels into system RAM. For each file, selects a single random slot among
10 windows in each group, shuffles groups, and trains one window per group.
"""

import argparse
import csv
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from model import build_wavenet

# TensorFlow normally reserves most GPU VRAM at initialization. The shared
# ligroup workstation enforces a 16 GiB per-user quota for the tidmad account.
# Enable on-demand allocation BEFORE creating the model or using the GPU.
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [
            tf.config.LogicalDeviceConfiguration(
                memory_limit=14336
            )
        ]
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
    parser.add_argument(
        "--max-windows-per-file",
        type=int,
        default=0,
        help="0 = all available; e.g. 20 for a smoke test",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save intermediate weights every N updates per file; 0 disables",
    )
    args = parser.parse_args()

    if args.end_file < args.start_file:
        parser.error("--end-file must be >= --start-file")
    if args.max_windows_per_file < 0:
        parser.error("--max-windows-per-file must be nonnegative")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_wavenet()
    model.summary()
    print("Trainable parameters:", model.count_params(), flush=True)

    csv_path = out / "training_log.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "step", "mse"])
        fh.flush()

        for file_num in range(args.start_file, args.end_file + 1):
            fname = Path(args.data_dir) / f"abra_training_{file_num:04d}.h5"
            if not fname.exists():
                raise FileNotFoundError(fname)

            # Match the original per-file Adam optimizer reset.
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

                rng = np.random.default_rng(args.seed + file_num)
                slot = int(rng.integers(0, SAMPLE_SIZE))
                indices = rng.permutation(num_groups)
                if args.max_windows_per_file:
                    indices = indices[: args.max_windows_per_file]

                print(
                    f"FILE {fname} slot={slot} batches={len(indices)}",
                    flush=True,
                )

                for step, group in enumerate(indices, start=1):
                    start = (int(group) * SAMPLE_SIZE + slot) * SEQ_LEN
                    x = np.asarray(
                        in_ds[start : start + SEQ_LEN], dtype=np.int16
                    ) + 128
                    y = np.asarray(
                        tar_ds[start : start + SEQ_LEN], dtype=np.float32
                    ) + 128.0

                    if x.shape[0] != SEQ_LEN or y.shape[0] != SEQ_LEN:
                        raise RuntimeError(
                            f"Short window in {fname} at sample {start}"
                        )
                    if x.min() < 0 or x.max() > 255:
                        raise ValueError(
                            f"ADC embedding input outside [0, 255] in {fname}"
                        )

                    loss = model.train_on_batch(
                        x.astype("int32")[None, :],
                        y[None, :, None],
                        return_dict=False,
                    )
                    loss = float(loss)
                    writer.writerow([file_num, step, loss])

                    if args.checkpoint_every and step % args.checkpoint_every == 0:
                        ckpt = out / (
                            f"WaveNet_MSE_FP32_file{file_num:04d}"
                            f"_step{step:05d}.weights.h5"
                        )
                        model.save_weights(ckpt)
                        fh.flush()
                        print(f"Checkpoint saved: {ckpt}", flush=True)

                    if step == 1 or step % 100 == 0 or step == len(indices):
                        print(
                            f" file={file_num} step={step}/{len(indices)} "
                            f"mse={loss:.6g}",
                            flush=True,
                        )
                        fh.flush()

            ckpt = out / f"WaveNet_MSE_FP32_after_{file_num:04d}.weights.h5"
            model.save_weights(ckpt)
            print("Saved:", ckpt, flush=True)

    final_ckpt = out / "WaveNet_MSE_FP32_final.weights.h5"
    model.save_weights(final_ckpt)
    print("Saved:", final_ckpt, flush=True)


if __name__ == "__main__":
    main()
