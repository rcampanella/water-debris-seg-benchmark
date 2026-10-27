#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-fix2-cross
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/fix2_yolo_crossdomain-%j.out
# Fairness fix 2: LaRS-trained YOLOv11s on FloW test / CANSURF val, debris-only + class-agnostic.   DATASET=flow|cansurf  WEIGHTS
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"; WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources"
WEIGHTS="${WEIGHTS:-${REPO}/eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/runs/lars_full_11s_stageB/weights/last.pt}"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
python "${WDS}/scripts/eval_yolo_crossdomain.py" --weights "${WEIGHTS}" --dataset "${DATASET:?}" \
  --out "${OUT}/fix2_yolo11s_lars_on_${DATASET}_${SLURM_JOB_ID}.json" 2>&1 | tee "${OUT}/fix2_yolo11s_lars_on_${DATASET}_${SLURM_JOB_ID}.log"
