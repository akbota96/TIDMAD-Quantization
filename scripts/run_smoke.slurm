#!/bin/bash
#SBATCH --job-name=wavenet-fp32-smoke
#SBATCH --account=ddp433
#SBATCH --partition=gpu-shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=fresh_fp32_smoke_%j.out
#SBATCH --error=fresh_fp32_smoke_%j.err
set -eo pipefail
source /cm/shared/apps/spack/0.17.3/gpu/b/opt/spack/linux-rocky8-skylake_avx512/gcc-8.5.0/anaconda3-2021.05-kfluefzsateihlamuk2qihp56exwe7si/etc/profile.d/conda.sh
conda activate qwavenet
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2
cd /expanse/lustre/projects/ddp433/aassan/TIDMAD/fresh_wavenet
python -u train.py --data-dir /expanse/lustre/projects/ddp433/aol002/Data --out-dir smoke_outputs --start-file 15 --end-file 15 --max-windows-per-file 20
