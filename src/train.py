"""Train fresh Keras FP32 MSE WaveNet on TIDMAD train files 0015..0019.

HDF5 read-on-demand avoids loading 2 billion ADC samples per channel into RAM.
The sampling structure matches the original script: one common random slot
out of 10 in each group of ten 40,000-sample windows, per training file.
"""
import argparse
import csv
import os
from pathlib import Path
import random
import h5py
import numpy as np
import tensorflow as tf
from model import build_wavenet

SEQ_LEN = 40000
SAMPLE_SIZE = 10
N_SAMPLES = 2_000_000_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', default='fresh_fp32_results')
    p.add_argument('--start-file', type=int, default=15)
    p.add_argument('--end-file', type=int, default=19)
    p.add_argument('--max-windows-per-file', type=int, default=0,
                   help='0 = all available; e.g. 20 for a smoke test')
    p.add_argument('--learning-rate', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=20261012)
    args = p.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build_wavenet()
    model.summary()
    print('Trainable parameters:', model.count_params(), flush=True)
    csv_path = out / 'training_log.csv'
    with csv_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['file', 'step', 'mse'])
        for file_num in range(args.start_file, args.end_file + 1):
            fname = Path(args.data_dir) / f'abra_training_{file_num:04d}.h5'
            if not fname.exists():
                raise FileNotFoundError(fname)
            # Original code recreates Adam for each training file.
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          loss='mse')
            with h5py.File(fname, 'r') as h5:
                in_ds = h5['timeseries/channel0001/timeseries']
                tar_ds = h5['timeseries/channel0002/timeseries']
                available = min(len(in_ds), len(tar_ds), N_SAMPLES)
                num_groups = available // (SEQ_LEN * SAMPLE_SIZE)
                if num_groups < 1:
                    raise RuntimeError(f'Not enough data in {fname}')
                rng = np.random.default_rng(args.seed + file_num)
                slot = int(rng.integers(0, SAMPLE_SIZE))
                indices = rng.permutation(num_groups)
                if args.max_windows_per_file:
                    indices = indices[:args.max_windows_per_file]
                print(f'FILE {fname} slot={slot} batches={len(indices)}', flush=True)
                for step, group in enumerate(indices, start=1):
                    start = (int(group) * SAMPLE_SIZE + slot) * SEQ_LEN
                    # Promote before +128, to avoid any signed-int8 ambiguity.
                    x = np.asarray(in_ds[start:start + SEQ_LEN], dtype=np.int16) + 128
                    y = np.asarray(tar_ds[start:start + SEQ_LEN], dtype=np.float32) + 128.0
                    if x.shape[0] != SEQ_LEN or y.shape[0] != SEQ_LEN:
                        raise RuntimeError(f'Incomplete sequence at {start}')
                    if x.min() < 0 or x.max() > 255:
                        raise ValueError('ADC outside valid embedding range')
                    loss = model.train_on_batch(x.astype('int32')[None, :],
                                                y[None, :, None], return_dict=False)
                    loss = float(loss)
                    writer.writerow([file_num, step, loss])
                    if step == 1 or step % 100 == 0 or step == len(indices):
                        print(f'  file={file_num} step={step}/{len(indices)} mse={loss:.6g}', flush=True)
                        fh.flush()
            ckpt = out / f'WaveNet_MSE_FP32_after_{file_num:04d}.weights.h5'
            model.save_weights(ckpt)
            print('Saved:', ckpt, flush=True)
    model.save_weights(out / 'WaveNet_MSE_FP32_final.weights.h5')


if __name__ == '__main__':
    main()
