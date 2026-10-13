#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH -J wdseg-ab07-loss
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_loss_smoke-%j.out

# One-batch loss check (patched vs upstream) before retraining VidEoMT on LaRS.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VID="${SCRATCH}/third_party/videomt"
WDS="${HOME}/projects/water-debris-seg-benchmark/water-debris-seg"
OUT="${WDS}/../eval_outputs/water_debris_seg/ablations/07_videomt_temporal/sources"
mkdir -p "${SCRATCH}/logs" "${OUT}"
source "${SCRATCH}/envs/eomt/bin/activate"
export PYTHONPATH="${VID}:${WDS}:${PYTHONPATH:-}"
cd "${VID}"
{
  echo "=== upstream (no keyframe patch) ==="
  python -u "${WDS}/scripts/videomt_one_batch_loss_smoke.py" --no-patch --batches 8 || true
  echo "=== patched ==="
  python -u "${WDS}/scripts/videomt_one_batch_loss_smoke.py" --batches 8
} 2>&1 | grep -v -i warn | tee "${OUT}/videomt_loss_smoke_${SLURM_JOB_ID}.log"
