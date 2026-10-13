#!/usr/bin/env python3
"""Keras FP32 WaveNet inference on TIDMAD validation files.

Preserve the original HDF5 schema and metadata by copying each validation file,
then replace only /timeseries/channel0001/timeseries with denoised int8 ADC values.
Channel 0002 and injection metadata are left unchanged.

--target-convention shifted: trained with SG target +128; subtract 128 at inference.
--target-convention signed:  trained with original signed SG target; no subtraction.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
import h5py
import tensorflow as tf
from model import build_wavenet

SEQ_LEN = 40_000

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [
            tf.config.LogicalDeviceConfiguration(
                memory_limit=10240
            )
        ]
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--start-file', type=int, default=15)
    parser.add_argument('--end-file', type=int, default=19)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Number of 40,000-sample windows per model call')
    parser.add_argument('--target-convention', choices=('shifted', 'signed'),
                        default='shifted',
                        help='MUST match target preprocessing in train.py')
    parser.add_argument('--dry-run', action='store_true',
                        help='Predict only first 2 windows per file; do not create output')
    args = parser.parse_args()
    if args.batch_size < 1 or args.start_file > args.end_file:
        parser.error('Invalid batch size or file range')
    if not args.weights.is_file():
        parser.error(f'Weights not found: {args.weights}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_wavenet(seq_len=SEQ_LEN)
    model.load_weights(str(args.weights))
    print(f'Loaded FP32 weights: {args.weights}; parameters={model.count_params()}', flush=True)
    print(f'Target convention: {args.target_convention}', flush=True)

    for file_num in range(args.start_file, args.end_file + 1):
        source = args.data_dir / f'abra_validation_{file_num:04d}.h5'
        dest = args.out_dir / f'abra_validation_denoised_wavenet_keras_mse_{file_num:04d}.h5'
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == dest.resolve():
            raise ValueError('Output must differ from input')
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite existing inference file: {dest}')

        with h5py.File(source, 'r') as src:
            squid = src['timeseries/channel0001/timeseries']
            sg = src['timeseries/channel0002/timeseries']
            total = squid.shape[0]
            if squid.ndim != 1 or sg.ndim != 1 or sg.shape[0] != total:
                raise ValueError(f'Unexpected channel shape in {source}')
            if total % SEQ_LEN:
                raise ValueError(f'{source}: {total} samples is not divisible by {SEQ_LEN}')
            if squid.dtype != np.dtype('int8'):
                raise ValueError(f'Expected signed int8 channel0001, got {squid.dtype}')
            n_windows = total // SEQ_LEN
            print(f'{source.name}: {total:,} samples, {n_windows:,} windows', flush=True)

            if args.dry_run:
                raw = squid[: 2 * SEQ_LEN]
                inp = raw.astype(np.int16).reshape(2, SEQ_LEN) + 128
                pred = model(inp.astype(np.int32), training=False).numpy().reshape(-1)
                offset = 128 if args.target_convention == 'shifted' else 0
                pred_signed = pred - offset
                print(f'  DRY RUN: input std={raw.std():.4f}, prediction std={pred_signed.std():.4f}, '
                      f'min={pred_signed.min():.3f}, max={pred_signed.max():.3f}; no output written', flush=True)
                continue

        # Copy entire HDF5 first: channel 2, attributes, and auxiliary groups remain intact.
        # This requires approximately one original validation file's worth of free disk space.
        print(f'Copying original HDF5 (including all metadata) -> {dest}', flush=True)
        shutil.copyfile(source, dest)
        try:
            with h5py.File(dest, 'r+') as dst:
                out_ds = dst['timeseries/channel0001/timeseries']
                offset = 128 if args.target_convention == 'shifted' else 0
                for start_window in range(0, n_windows, args.batch_size):
                    count = min(args.batch_size, n_windows - start_window)
                    lo = start_window * SEQ_LEN
                    hi = lo + count * SEQ_LEN
                    raw = np.asarray(out_ds[lo:hi], dtype=np.int16)
                    inp = (raw + 128).reshape(count, SEQ_LEN).astype(np.int32)
                    pred = model(inp, training=False).numpy()
                    if pred.shape != (count, SEQ_LEN, 1):
                        raise RuntimeError(f'Unexpected prediction shape: {pred.shape}')
                    pred = pred.reshape(-1).astype(np.float32) - offset
                    if not np.isfinite(pred).all():
                        raise FloatingPointError(f'Nonfinite prediction in file {file_num} window {start_window}')
                    # Avoid int8 modular wraparound. Preserve nearest-integer ADC counts.
                    quantized = np.clip(np.rint(pred), -128, 127).astype(np.int8)
                    out_ds[lo:hi] = quantized
                    if start_window == 0 or (start_window + count) % 1000 < count:
                        print(f'  processed {start_window + count:,}/{n_windows:,} windows', flush=True)
                dst.attrs['denoising_model'] = 'WaveNet_MSE_FP32_Keras'
                dst.attrs['model_output_target_convention'] = args.target_convention
                dst.attrs['inference_weights_filename'] = args.weights.name
                dst.flush()
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        print(f'COMPLETE: {dest}', flush=True)


if __name__ == '__main__':
    main()
