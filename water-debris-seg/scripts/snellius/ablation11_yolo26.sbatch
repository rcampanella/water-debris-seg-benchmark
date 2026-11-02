#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-ab11-yolo26
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation11_yolo26-%j.out

# Ablation 11: one YOLO26 / YOLO26-seg model trained and tested on one dataset (train_yolo26.py: test split monitored
# every epoch, best.pt, latency). Detection models are also scored with pycocotools (eval_yolo_lars.py; FloW / CANSURF
# with the main-table GT, LaRS with the ablation 10 box GT, crowd ignored) for reference. DATASET=lars11 (11 classes
# with Water / Sky) is scored with LaRS PQ instead (eval_sam_panoptic.py on best.pt and last.pt: detection boxes ->
# SAM masks, seg models' own masks; confidence thresholds PQ_THRESHOLDS), and best.pt at confidence 0.20 also with
# the official LaRS evaluator (maps exported as LaRS PNGs, run_lars_evaluator.py; results in <ablation>/lars_official).
#   DATASET=lars|lars11|flow|cansurf   MODEL=yolo26{n,s,m,l,x}[-seg]   EPOCHS (default 50)   FRACTION / NAME (smoke tests)
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
ABL="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family"
DATASET="${DATASET:?set DATASET=lars|lars11|flow|cansurf}"
MODEL="${MODEL:?set MODEL=yolo26n|...|yolo26x-seg}"
NAME="${NAME:-${DATASET}_${MODEL}}"
J="${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}/logs" "${ABL}/sources"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
export WANDB_RUN_GROUP="ablation11_yolo26"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "${WDS}/scripts/train_yolo26.py" --dataset "${DATASET}" --model "${MODEL}" --epochs "${EPOCHS:-50}" \
  --fraction "${FRACTION:-1.0}" --name "${NAME}" 2>&1 | tee "${ABL}/sources/train_${NAME}_${J}.log"

case "${DATASET}" in
  flow)    GT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json"
           IMAGES="${REPO}/data/water_debris/yolo_splits/flow/test/images" ;;
  cansurf) GT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
           IMAGES="${REPO}/data/water_debris/cansurf_home/val/images" ;;
  lars11)  GT="" ;;
  lars)    GT="${REPO}/eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/sources/lars_val_gt_coco_boxes.json"
           IMAGES="${SCRATCH}/data/lars/val/images" ;;
esac
[[ "${MODEL}" == *-seg ]] && GT=""   # seg models: Ultralytics box + mask metrics only
if [[ -n "${GT}" ]]; then
  python "${WDS}/scripts/eval_yolo_lars.py" --weights "${ABL}/runs/${NAME}/weights/best.pt" --gt "${GT}" \
    --images "${IMAGES}" --data "${ABL}/data/${DATASET}_detect.yaml" \
    --out "${ABL}/sources/coco_eval_${NAME}_${J}.json" 2>&1 | tee "${ABL}/sources/coco_eval_${NAME}_${J}.log"
fi
if [[ "${DATASET}" == lars11 ]]; then
  SRC=yolo; [[ "${MODEL}" == *-seg ]] && SRC=yolo-seg
  for CK in best last; do
    # shellcheck disable=SC2086
    PYTHONPATH="${WDS}/scripts:${WDS}:${REPO}/data/water_debris/third_party/eomt:${PYTHONPATH}" \
      python "${WDS}/scripts/eval_sam_panoptic.py" --source "${SRC}" --weights "${ABL}/runs/${NAME}/weights/${CK}.pt" \
      --split val --thresholds ${PQ_THRESHOLDS:-0.1 0.2 0.3 0.4 0.5} --out "${ABL}/sources/pq_${NAME}_${CK}_${J}.json" \
      2>&1 | tee "${ABL}/sources/pq_${NAME}_${CK}_${J}.log"
  done
  PRED="${REPO}/data/water_debris/lars_eval/preds/ab11_${NAME}"
  rm -rf "${PRED}"
  PYTHONPATH="${WDS}/scripts:${WDS}:${REPO}/data/water_debris/third_party/eomt:${PYTHONPATH}" \
    python "${WDS}/scripts/eval_sam_panoptic.py" --source "${SRC}" --weights "${ABL}/runs/${NAME}/weights/best.pt" \
    --split val --thresholds 0.20 --export-png "${PRED}" --out "${ABL}/sources/lars_official_scorer_ab11_${NAME}_${J}.json"
  "${HOME}/envs/lars-eval/bin/python" "${WDS}/scripts/run_lars_evaluator.py" --method "ab11_${NAME}" \
    --scorer-json "${ABL}/sources/lars_official_scorer_ab11_${NAME}_${J}.json" --out-dir "${ABL}"
fi
echo "${NAME} done"
