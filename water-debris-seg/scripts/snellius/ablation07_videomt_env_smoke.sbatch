#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH -J wdseg-ab07-smoke
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_env_smoke-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${SCRATCH}/envs/eomt/bin/activate"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/videomt_env_smoke.py" \
  | tee "${OUT}/sources/videomt_env_smoke_${SLURM_JOB_ID}.log"
echo "Ablation 07 env smoke finished (exit code from script above)."
