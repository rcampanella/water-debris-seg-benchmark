#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-sam-panoptic
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/sam_panoptic-%j.out

# Fairness fixes 1 (EoMT + SAM control) and 2 (YOLOv11s + SAM): detections -> SAM masks -> LaRS PQ / mIoU.
#   SOURCE=yolo|eomt   WEIGHTS   SPLIT=held10|val   THRESHOLDS ("0.05 0.1 ...", yolo)   NOSAM=1 (check)   LIMIT   TAG
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
EOMT="${REPO}/data/water_debris/third_party/eomt"
OUT="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources"
SOURCE="${SOURCE:?}"; WEIGHTS="${WEIGHTS:?}"; SPLIT="${SPLIT:?}"
NAME="sam_panoptic_${SOURCE}_${SPLIT}${TAG:+_${TAG}}"
mkdir -p "/scratch-shared/${USER}/water-debris-seg/logs" "${OUT}"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
EXTRA=(); [[ -n "${NOSAM:-}" ]] && EXTRA+=(--no-sam)
# shellcheck disable=SC2086
python "${WDS}/scripts/eval_sam_panoptic.py" --source "${SOURCE}" --weights "${WEIGHTS}" --split "${SPLIT}" \
  --thresholds ${THRESHOLDS:-0.25} --limit "${LIMIT:-0}" "${EXTRA[@]}" --out "${OUT}/${NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee "${OUT}/${NAME}_${SLURM_JOB_ID}.log"
