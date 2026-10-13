#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab08-yolo-flow
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation08_yolo_flow-%j.out

# YOLOv11n on FloW-Img: same recipe as ablation08_yolo_cansurf.sbatch.
# Train on FloW training/ minus a seeded 10% val slice; score on the official test split
# with eval_coco_bbox.py against the ablation 04 GT so mAP/AP_small match EoMT mask→box.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
FLOW="${SCRATCH}/data/flow"
DATA="${FLOW}/yolo"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/08_specialized_yolo"
GT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json"
IMGSZ="${IMGSZ:-640}"
NAME="${NAME:-flow_train}"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation08_yolo_flow}"
export WANDB_NAME="${WANDB_NAME:-yolo11n_flow_${SLURM_JOB_ID}}"
export WANDB_PROJECT="${WANDB_PROJECT:-water-debris-seg}"
export YOLO_CONFIG_DIR="${SCRATCH}/artifacts/ultralytics_config"
mkdir -p "${YOLO_CONFIG_DIR}"

[[ -f "${DATA}/data.yaml" ]] || python "${WDS}/scripts/prepare_flow_yolo.py" --flow-root "${FLOW}" --out "${DATA}"
cp "${DATA}/data.yaml" "${OUT}/sources/flow_data.yaml"
cp "${DATA}/val_split.txt" "${OUT}/sources/flow_val_split.txt"
yolo settings wandb=True 2>/dev/null || true

yolo train \
  task=detect \
  data="${DATA}/data.yaml" \
  model=yolo11n.pt \
  epochs=50 \
  imgsz="${IMGSZ}" \
  batch=16 \
  device=0 \
  project="${OUT}" \
  name="${NAME}" \
  exist_ok=True \
  2>&1 | tee "${OUT}/sources/flow_train_${SLURM_JOB_ID}.log"

WEIGHTS="${OUT}/${NAME}/weights/best.pt"

# Ultralytics metrics on the test split (mAP50-95 / mAP50 / P / R)
yolo val \
  task=detect \
  model="${WEIGHTS}" \
  data="${DATA}/data.yaml" \
  split=test \
  imgsz="${IMGSZ}" \
  device=0 \
  project="${OUT}" \
  name="${NAME}_test" \
  exist_ok=True \
  2>&1 | tee "${OUT}/sources/flow_test_val_${SLURM_JOB_ID}.log"

# Same scorer and GT as ablation 04 (mAP, AP50, AP75, AP_small)
PRED="${OUT}/sources/yolo11n_flow_test_preds_coco.json"
python "${WDS}/scripts/yolo_predict_to_coco.py" \
  --weights "${WEIGHTS}" --gt-coco "${GT}" --images "${FLOW}/test/images" \
  --out "${PRED}" --imgsz "${IMGSZ}"
python "${WDS}/scripts/eval_coco_bbox.py" \
  --gt-coco "${GT}" --pred-coco "${PRED}" \
  --out "${PRED%.json}.metrics.json" \
  2>&1 | tee "${OUT}/sources/flow_test_cocoeval_${SLURM_JOB_ID}.log"

echo "Ablation 08 YOLO FloW done."
