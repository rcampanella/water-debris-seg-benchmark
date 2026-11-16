#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J wdseg-depth-prep
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_prepare-%j.out

# Depth benchmark D01, step 4 prep: Ultralytics depth datasets (standard split D6, unadapted raw GT, D10 frames only).
# Output (home, survives the scratch purge, README D8): data/water_debris/depth_d01/<source>/ + <source>.yaml
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
SPLIT="${REPO}/eval_outputs/water_debris_depth/ablations/D01_rgb/sources/gt_unadapted_v1"
OUT="${REPO}/data/water_debris/depth_d01"
GT="/scratch-shared/${USER}/water-debris-seg/depth_gt/unadapted_v1"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
cd /scratch-shared/${USER}/water-debris-seg
for src in multiaqua pohang; do
  python -I "${REPO}/water-debris-seg/scripts/depth/prepare_depth_dataset.py" --source "$src" \
    --manifest "${SPLIT}/manifest_standard_${src}.json" --gt "${GT}" --out "${OUT}" --workers 16
done
du -sh "${OUT}"/*
echo "done prepare"
