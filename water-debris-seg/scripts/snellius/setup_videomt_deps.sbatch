#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-videomt-dep
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/setup_videomt_deps-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

module load 2025
module load Python/3.13.5-GCCcore-14.3.0

bash "${WDS}/scripts/install_videomt_deps.sh" \
  2>&1 | tee "${OUT}/sources/videomt_deps_${SLURM_JOB_ID}.log"

echo "setup_videomt_deps done."
