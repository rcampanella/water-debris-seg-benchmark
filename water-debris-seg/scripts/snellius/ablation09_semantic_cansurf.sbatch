#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab09-sem
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation09_semantic_cansurf-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/09_cross_dataset_transfer"
GT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco.json"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/mk2w7dr8/checkpoints/epoch=30-step=2356.ckpt"
PRED="${OUT}/sources/eomt_lars_semantic_cansurf_preds_coco.json"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation09_semantic_cansurf}"
export WANDB_NAME="${WANDB_NAME:-eomt_lars_semantic_cansurf_${SLURM_JOB_ID}}"

python "${WDS}/scripts/eval_eomt_cansurf_semantic_mask2box.py" \
  --gt-coco "${GT}" \
  --lightning-ckpt "${CKPT}" \
  --out "${PRED}" \
  --wandb \
  --wandb-group "${WANDB_RUN_GROUP}" \
  --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/semantic_cansurf_${SLURM_JOB_ID}.log"

echo "Ablation 09 semantic → CANSURF done."
