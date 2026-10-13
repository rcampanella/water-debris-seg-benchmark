#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -J wdseg-ab05-gt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation05_cansurf_gt-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
CANSURF="${SCRATCH}/data/cansurf"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${SCRATCH}/envs/eomt/bin/activate" 2>/dev/null || true
python "${WDS}/scripts/masks_to_coco_boxes.py" yolo-split \
  --images "${CANSURF}/val/images" \
  --labels "${CANSURF}/val/labels" \
  --out "${OUT}/sources/cansurf_val_gt_coco.json"
echo "CANSURF val GT COCO JSON -> ${OUT}/sources/cansurf_val_gt_coco.json"
