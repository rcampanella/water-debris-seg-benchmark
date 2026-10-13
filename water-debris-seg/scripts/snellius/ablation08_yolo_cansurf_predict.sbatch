#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -J wdseg-ab08-yolo-cansurf-pred
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation08_yolo_cansurf_predict-%j.out

# Writes COCO predictions from the existing CANSURF YOLO checkpoint (train job 27648816) so it
# is scored by eval_coco_bbox.py (pycocotools) on the same aligned GT as EoMT. No training.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/08_specialized_yolo"
GT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
PRED="${OUT}/sources/yolo11n_cansurf_val_preds_coco.json"

# shellcheck disable=SC1091
source "${SCRATCH}/envs/eomt/bin/activate"
export YOLO_CONFIG_DIR="${SCRATCH}/artifacts/ultralytics_config"
python "${WDS}/scripts/yolo_predict_to_coco.py" \
  --weights "${OUT}/cansurf_train/weights/best.pt" --gt-coco "${GT}" \
  --images "${SCRATCH}/data/cansurf/val/images" --out "${PRED}" --imgsz 640
python "${WDS}/scripts/eval_coco_bbox.py" --gt-coco "${GT}" --pred-coco "${PRED}" \
  --out "${PRED%.json}.metrics.json" 2>&1 | tee "${OUT}/sources/cansurf_cocoeval_${SLURM_JOB_ID}.log"
