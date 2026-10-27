#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH -J wdseg-ab05-gt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation05_eval_yolo_aligned-%j.out

set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object"
GT="${OUT}/sources/cansurf_val_gt_coco_yolo_aligned.json"
mkdir -p "/scratch-shared/${USER}/water-debris-seg/logs"

source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${PYTHONPATH:-}"

run_eval() {
  local tag="$1" pred="$2" out="$3"
  python "${WDS}/scripts/eval_coco_bbox.py" \
    --gt-coco "${GT}" --pred-coco "${pred}" --out "${out}"
  echo "=== ${tag} (YOLO-aligned GT) ==="
  cat "${out}"
}

run_eval zeroshot "${OUT}/sources/eomt_zeroshot_val_preds_coco.json" \
  "${OUT}/sources/eomt_zeroshot_yolo_aligned_gt.metrics.json"
run_eval lars_panoptic "${OUT}/sources/eomt_lars_panoptic_ft_val_preds_coco.json" \
  "${OUT}/sources/eomt_lars_panoptic_yolo_aligned_gt.metrics.json"

echo "YOLO-aligned GT re-score done."
