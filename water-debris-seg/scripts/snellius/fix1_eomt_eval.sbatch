#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-fix1-eomt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/fix1_eomt_eval-%j.out

# Fairness fixes 1 / 1b: EoMT PQ, mIoU, box AP on LaRS val; debris-only + class-agnostic box AP on FloW / CANSURF.
#   MODEL=coco133|lars_panoptic|lars_semantic   DATASET=lars|flow|cansurf   MAPPING=tightened|full (coco133)
#   CKPT (override)   LIMIT (smoke tests)   TAG (extra name part)
# Code, checkpoints and the venv are read from home (the 2026-10-23 scratch purge deleted re-created files).
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
EOMT="${REPO}/data/water_debris/third_party/eomt"     # = upstream 0f1e1cc (water-debris-seg/envs/third_party_commits.txt)
CK="${REPO}/data/water_debris/checkpoints"
OUT="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources"
MODEL="${MODEL:?set MODEL}"; DATASET="${DATASET:?set DATASET}"; MAPPING="${MAPPING:-tightened}"
case "${MODEL}" in
  coco133)       CKPT="${CKPT:-${CK}/eomt_coco/coco_panoptic_eomt_base_640_2x.bin}" ;;
  lars_panoptic) CKPT="${CKPT:-${CK}/ab03_panoptic_67kk33hk/epoch=11-step=7812.ckpt}" ;;
  lars_semantic) CKPT="${CKPT:-${CK}/ab02_semantic_mk2w7dr8/epoch=30-step=2356.ckpt}" ;;
  box_instance)  CKPT="${CKPT:?set CKPT for box_instance}" ;;
esac
NAME="fix1_${MODEL}_${DATASET}${TAG:+_${TAG}}"
[[ "${MODEL}" == "coco133" ]] && NAME="${NAME}_map-${MAPPING}"
mkdir -p "${SCRATCH}/logs" "${OUT}"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
python "${WDS}/scripts/eval_eomt_fairness.py" --model "${MODEL}" --ckpt "${CKPT}" --dataset "${DATASET}" \
  --mapping "${MAPPING}" --limit "${LIMIT:-0}" --out "${OUT}/${NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee "${OUT}/${NAME}_${SLURM_JOB_ID}.log"
