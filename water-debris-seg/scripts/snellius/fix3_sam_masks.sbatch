#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-fix3-sam
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/fix3_sam_masks-%j.out
# Fairness fix 3, step 1-2: SAM pseudo-masks for FloW / CANSURF training images + quality check.   DATASET=flow|cansurf  LIMIT
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"; WDS="${REPO}/water-debris-seg"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
python "${WDS}/scripts/make_sam_masks.py" --dataset "${DATASET:?}" --limit "${LIMIT:-0}" 2>&1 | tee "${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources/sam_masks_${DATASET}_${SLURM_JOB_ID}.log"
