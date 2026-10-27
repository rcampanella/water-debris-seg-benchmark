#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH -J wdseg-ab07-inv
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_lars_seq_inventory-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/inventory_lars_sequences.py" \
  --out "${OUT}/sources/lars_sequences_inventory.json" \
  | tee "${OUT}/sources/lars_seq_inventory_${SLURM_JOB_ID}.log"

GT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco.json"
GT_ALIGNED="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
python "${WDS}/scripts/filter_coco_gt_images.py" \
  --gt-coco "${GT}" \
  --blocklist "${WDS}/data/cansurf_corrupt_val_images.txt" \
  --out "${GT_ALIGNED}"

echo "Ablation 07 inventory + CANSURF aligned GT done."
