#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab10-yolo-lars
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation10_yolo_lars-%j.out

# Ablation 10: YOLOv11n on LaRS boxes traced from the panoptic masks (see ablations/10_yolo_lars_boxes/TRAINING.md).
#   VARIANT=full|lora   STAGE=A (train90, held10 monitored; picks epochs) | B (full train, last.pt, then scored on LaRS val)
#   LOSS=bce|focal (default bce; focal runs are named lars_<variant>_focal_stage<STAGE>)
#   TAG (overrides the <variant>[_focal] part of run names)   EXTRA (extra train_yolo_lars.py args, e.g. "--lr0 0.00769")
#   EPOCHS (stage A budget, default 100)   FRACTION (smoke tests)   NAME (run name override)
#   DATASET=lars|flow|cansurf (default lars; flow/cansurf = fairness fix 3, outputs in fairness_fixes/, test = FloW test / CANSURF val)
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATASET="${DATASET:-lars}"
case "${DATASET}" in
  lars)    ABL="${REPO}/eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes"; GROUP="ablation10_yolo_lars"
           GT="${ABL}/sources/lars_val_gt_coco_boxes.json"; IMAGES="${SCRATCH}/data/lars/val/images"
           EVAL_YAML="${SCRATCH}/data/lars/yolo_boxes/eval_val.yaml" ;;
  flow)    ABL="${REPO}/eval_outputs/water_debris_seg/fairness_fixes"; GROUP="fix3_sam_box_datasets"
           GT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json"
           IMAGES="${REPO}/data/water_debris/yolo_splits/flow/test/images"; EVAL_YAML="${REPO}/data/water_debris/yolo_splits/flow/eval_test.yaml" ;;
  cansurf) ABL="${REPO}/eval_outputs/water_debris_seg/fairness_fixes"; GROUP="fix3_sam_box_datasets"
           GT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
           IMAGES="${REPO}/data/water_debris/cansurf_home/val/images"; EVAL_YAML="${REPO}/data/water_debris/yolo_splits/cansurf/eval_test.yaml" ;;
  *) echo "unknown DATASET ${DATASET}"; exit 1 ;;
esac
VARIANT="${VARIANT:?set VARIANT=full|lora}"
STAGE="${STAGE:?set STAGE=A|B}"
LOSS="${LOSS:-bce}"
if [[ -z "${TAG:-}" ]]; then TAG="${VARIANT}"; [[ "${LOSS}" == "focal" ]] && TAG="${VARIANT}_focal"; fi
NAME="${NAME:-${DATASET}_${TAG}_stage${STAGE}}"
mkdir -p "${SCRATCH}/logs" "${ABL}/sources"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"   # on home since 2026-10-23 (scratch purge)
export WANDB_RUN_GROUP="${GROUP}"
export LORA_R="${LORA_R:-8}" LORA_ALPHA="${LORA_ALPHA:-16}"

ARGS=(--dataset "${DATASET}" --variant "${VARIANT}" --stage "${STAGE}" --loss "${LOSS}" --tag "${TAG}" --name "${NAME}" --fraction "${FRACTION:-1.0}")
if [[ "${STAGE}" == "A" ]]; then
  ARGS+=(--epochs "${EPOCHS:-100}")
else
  ARGS+=(--epochs-from "${ABL}/runs/${DATASET}_${TAG}_stageA/results.csv")
fi
# shellcheck disable=SC2086
python "${WDS}/scripts/train_yolo_lars.py" "${ARGS[@]}" ${EXTRA:-} 2>&1 | tee "${ABL}/sources/train_${NAME}_${SLURM_JOB_ID}.log"

if [[ "${STAGE}" == "B" ]]; then
  python "${WDS}/scripts/eval_yolo_lars.py" \
    --weights "${ABL}/runs/${NAME}/weights/last.pt" \
    --gt "${GT}" --images "${IMAGES}" --data "${EVAL_YAML}" \
    --out "${ABL}/sources/eval_${NAME}_${SLURM_JOB_ID}.json" 2>&1 | tee "${ABL}/sources/eval_${NAME}_${SLURM_JOB_ID}.log"
fi
echo "${DATASET} ${VARIANT} stage ${STAGE} done"
