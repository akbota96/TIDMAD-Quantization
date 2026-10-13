FRESH WAVENET MSE FP32 -- experiment 0

1. Copy model.py, train.py, run_smoke.slurm into:
   /expanse/lustre/projects/ddp433/aassan/TIDMAD/fresh_wavenet

2. On Expanse login, run:
   cd /expanse/lustre/projects/ddp433/aassan/TIDMAD/fresh_wavenet
   sbatch run_smoke.slurm

3. After it finishes, inspect:
   tail -40 fresh_fp32_smoke_<JOB_ID>.out
   tail -40 fresh_fp32_smoke_<JOB_ID>.err

Do not launch full training until smoke run succeeds and the FP32
architecture/input-output behavior is validated.

The original network.py has: embedding 256x16; 10 dilated blocks;
gate/filter convs; skip and residual 1x1 convs; and two final 1x1
convs. This model changes only the last output width 256 -> 1 for
MSE regression, keeping raw shifted ADC input and regression target.

Expected parameter count for these defaults: 294,369 (not 293,313).
The older model's count 293,313 is 1,056 fewer, equal to one
32-to-32 1x1 convolution with bias. Check exact difference in original
models before asserting architecture identity or performance parity.

This script uses original per-file Adam reset, LR 5e-4, random slot
per file, 5,000 groups/file, and shuffles groups. Its RNG is seeded.
20-window smoke mode is NOT the full training regime.

INFERENCE (after FULL FP32 training finishes)
------------------------------------------------
Dry-run first (no output file written):
python infer.py --data-dir /expanse/lustre/projects/ddp433/aol002/Data \
  --weights fresh_fp32_results/WaveNet_MSE_FP32_final.weights.h5 \
  --out-dir fresh_fp32_inference --start-file 15 --end-file 15 --dry-run

Full inference for validation 0015 through 0019:
python infer.py --data-dir /expanse/lustre/projects/ddp433/aol002/Data \
  --weights fresh_fp32_results/WaveNet_MSE_FP32_final.weights.h5 \
  --out-dir fresh_fp32_inference --start-file 15 --end-file 19 --batch-size 4 \
  --target-convention shifted

IMPORTANT: 'shifted' matches the CURRENT train.py (target = SG + 128).
If you change train.py to use signed SG targets, run inference with
--target-convention signed INSTEAD, otherwise results will have a 128-count error.
Run on a GPU compute node; do not perform TensorFlow inference on the login node.
Outputs are complete copies of the original HDF5 with only channel0001's
samples changed (and a few root provenance attributes added).
Requires enough disk space for copies of each multi-GB validation file.
