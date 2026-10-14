#!/usr/bin/env python3
"""TIDMAD Benchmark 1 (Denoising Score), with explicit file selection and audit logs.

The PSD, SNR, SG-normalization, and base-5.27 log score follow the supplied
TIDMAD benchmark implementation. By default, scores are computed jointly over
all selected files (with SG normalization across all selected seconds).
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

SAMPLES_PER_SECOND = 10_000_000
BASE = 5.27


def one_second_psd(path, channel, second):
    start = second * SAMPLES_PER_SECOND
    with h5py.File(path, 'r') as h5:
        group = h5[f'timeseries/channel{channel:04d}']
        ds = group['timeseries']
        data = ds[start:start + SAMPLES_PER_SECOND]
        voltage_range = float(h5['timeseries/channel0001'].attrs['voltage_range_mV'])
        sampling_frequency = float(h5['timeseries/channel0001'].attrs['sampling_frequency'])
    if len(data) != SAMPLES_PER_SECOND:
        raise ValueError(f'Incomplete one-second chunk at second {second} in {path}')
    ts = np.asarray(data, dtype=np.float32) * np.float32(voltage_range / (2 * 128.0))
    delta_t = 1.0 / sampling_frequency
    # Original code: dt/N * abs(rfft(TS))**2, discarding DC.
    pwr = (delta_t / SAMPLES_PER_SECOND) * np.abs(np.fft.rfft(ts))[1:] ** 2
    # Retain the original benchmark's frequency-grid convention exactly.
    freq = np.linspace(0, 5e6, SAMPLES_PER_SECOND // 2)
    return freq, pwr


def peak_index(power):
    if len(power) < 3:
        raise ValueError('PSD too short to locate signal peak')
    differences = power[1:-1] - power[:-2] - power[2:]
    return int(np.argmax(differences)) + 1


def snr_at_peak(freq, power, index):
    if index < 50 or index + 50 >= len(power):
        raise ValueError(f"Peak index {index} too close to PSD boundary")

    signal = np.sum(power[index - 1:index + 2], dtype=np.float64)

    noise_left = np.sum(power[index - 50:index - 1], dtype=np.float64)
    noise_right = np.sum(power[index + 2:index + 51], dtype=np.float64)
    noise = noise_left + noise_right

    if noise <= 0 or not np.isfinite(noise):
        raise ValueError(f"Invalid noise estimate: {noise}")

    return float(signal / noise), float(freq[index])

def score_one_second(work):
    file_name, second = work
    sg_freq, sg_psd = one_second_psd(file_name, 2, second)
    center = peak_index(sg_psd)
    sg_snr, _ = snr_at_peak(sg_freq, sg_psd, center)
    squid_freq, squid_psd = one_second_psd(file_name, 1, second)
    squid_snr, _ = snr_at_peak(squid_freq, squid_psd, center)
    return file_name, second, sg_snr, squid_snr


def benchmark_score(rows):
    sg = np.asarray([r[2] for r in rows], dtype=np.float64)
    squid = np.asarray([r[3] for r in rows], dtype=np.float64)
    if not len(sg) or not np.isfinite(sg).all() or not np.isfinite(squid).all():
        raise ValueError('Empty or non-finite SNR measurements')
    max_sg = float(np.max(sg))
    if max_sg <= 0:
        raise ValueError('Non-positive SG maximum SNR; cannot normalize')
    weighted_mean = float(np.mean((sg / max_sg) * squid))
    # Exactly follows provided benchmark: round to 2 decimals, add 1e-10,
    # then take the logarithm base 5.27.
    rounded = float(np.round(weighted_mean, decimals=2)) + 1e-10
    if rounded <= 0:
        raise ValueError(f'Non-positive weighted SNR after rounding: {rounded}')
    return math.log(rounded, BASE), weighted_mean


def main():
    parser = argparse.ArgumentParser(description='Evaluate selected HDF5 files using the original TIDMAD Benchmark 1 score.')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory containing HDF5 files')
    parser.add_argument('--files', nargs='+', required=True, help='Exact HDF5 filenames in --input-dir (not a glob)')
    parser.add_argument('--coarse', action='store_true', help='Evaluate seconds 0,10,20,... per file; otherwise every second')
    parser.add_argument('--workers', type=int, default=1, help='Parallel CPU workers (default 1; 4-8 for faster fine scans)')
    parser.add_argument('--log-file', type=Path, default=Path('benchmark_results.log'), help='Append-only human-readable results log')
    parser.add_argument('--json-file', type=Path, default=Path('benchmark_results.json'), help='Machine-readable result file, overwritten on each run')
    parser.add_argument('--seconds-csv', type=Path, default=None, help='Optional CSV with each evaluated second and raw SNRs')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    root = args.input_dir.expanduser().resolve(strict=True)
    selected = []
    tasks = []
    for name in args.files:
        if Path(name).name != name or not name.endswith('.h5'):
            parser.error(f'--files expects plain .h5 filenames inside input directory: {name!r}')
        path = (root / name).resolve(strict=True)
        if path.parent != root:
            parser.error(f'File must be located directly in --input-dir: {name}')
        if path in selected:
            parser.error(f'Duplicate file specified: {name}')
        with h5py.File(path, 'r') as h5:
            one = h5['timeseries/channel0001/timeseries']
            two = h5['timeseries/channel0002/timeseries']
            if one.ndim != 1 or two.ndim != 1 or len(one) != len(two):
                raise ValueError(f'Invalid channel lengths in {path}')
            seconds = len(one) // SAMPLES_PER_SECOND
        if not seconds:
            raise ValueError(f'{path} is shorter than one second')
        # The reference implementation uses 200 seconds per file for a list,
        # but uses all complete seconds when given a single file.
        if len(args.files) > 1:
            if seconds < 200:
                raise ValueError(f'{path}: multi-file reference benchmark requires 200 full seconds')
            seconds = 200
        selected.append(path)
        tasks.extend((str(path), sec) for sec in range(0, seconds, 10 if args.coarse else 1))
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f'Input directory: {root}', flush=True)
    print(f'Mode: {"Coarse" if args.coarse else "Fine"}; files: {len(selected)}; evaluated seconds: {len(tasks)}', flush=True)
    for p in selected:
        print(f'  {p.name}', flush=True)
    rows = []
    if args.workers == 1:
        for i, task in enumerate(tasks, 1):
            rows.append(score_one_second(task))
            if i % 10 == 0 or i == len(tasks):
                print(f'  processed {i}/{len(tasks)} seconds', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(score_one_second, task) for task in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                rows.append(f.result())
                if i % 10 == 0 or i == len(tasks):
                    print(f'  processed {i}/{len(tasks)} seconds', flush=True)
    order = {str(path): i for i, path in enumerate(selected)}
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    overall, weighted = benchmark_score(rows)
    by_file = []
    for path in selected:
        group = [r for r in rows if r[0] == str(path)]
        score, wmean = benchmark_score(group)
        by_file.append({'file': path.name, 'seconds_evaluated': len(group), 'score': score, 'weighted_mean': wmean})
    report = {
        'timestamp_utc': timestamp,
        'input_directory': str(root),
        'files': [p.name for p in selected],
        'mode': 'Coarse' if args.coarse else 'Fine',
        'workers': args.workers,
        'seconds_evaluated': len(rows),
        'score': overall,
        'weighted_mean_before_rounding': weighted,
        'per_file': by_file,
        'score_definition': 'log_base_5.27(round(mean((SG_SNR / max_SG_SNR) * channel1_SNR), 2) + 1e-10)',
    }
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open('a', encoding='utf-8') as log:
        log.write(f'[{timestamp}] {report["mode"]} TIDMAD Benchmark 1\n')
        log.write(f'  Input directory: {root}\n')
        for item in by_file:
            log.write(f'  File: {item["file"]} | seconds={item["seconds_evaluated"]} | standalone_score={item["score"]:.8f}\n')
        log.write(f'  Files evaluated: {len(selected)} | seconds={len(rows)}\n')
        log.write(f'  COMBINED {report["mode"]} DENOISING SCORE: {overall:.8f}\n\n')
    args.json_file.parent.mkdir(parents=True, exist_ok=True)
    args.json_file.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.seconds_csv is not None:
        args.seconds_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.seconds_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'second', 'snr_sg', 'snr_channel1'])
            writer.writerows((Path(name).name, second, sg, squid) for name, second, sg, squid in rows)
    print(f'COMBINED {report["mode"]} DENOISING SCORE: {overall:.8f}', flush=True)
    print(f'Log: {args.log_file} | JSON: {args.json_file}', flush=True)


if __name__ == '__main__':
    main()
