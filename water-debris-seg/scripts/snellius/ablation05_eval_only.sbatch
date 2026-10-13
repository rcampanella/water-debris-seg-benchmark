#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -J wdseg-ab05-eval
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation05_eval-%j.out

set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object"
GT="${OUT}/sources/cansurf_val_gt_coco.json"
PREDS="${OUT}/sources/eomt_zeroshot_val_preds_coco.json"
VENV="/scratch-shared/${USER}/water-debris-seg/envs/eomt"
mkdir -p "/scratch-shared/${USER}/water-debris-seg/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation05_eomt_cansurf_zeroshot}"
export WANDB_NAME="${WANDB_NAME:-eomt_cansurf_mask2box_eval_${SLURM_JOB_ID}}"

python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
  --gt-coco "${GT}" \
  --pred-coco "${PREDS}" \
  --out "${PREDS}" \
  --wandb \
  --wandb-group "${WANDB_RUN_GROUP}" \
  --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/eomt_eval_${SLURM_JOB_ID}.log"
