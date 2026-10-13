#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab05-eomt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation05_eomt_cansurf-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object"
GT="${OUT}/sources/cansurf_val_gt_coco.json"
CKPT="${SCRATCH}/artifacts/checkpoints/eomt/coco_panoptic_eomt_base_640_2x.bin"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation05_eomt_cansurf_zeroshot}"
export WANDB_NAME="${WANDB_NAME:-eomt_cansurf_mask2box_${SLURM_JOB_ID}}"

bash "${WDS}/scripts/download_eomt_ckpt.sh"

python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
  --gt-coco "${GT}" \
  --out "${OUT}/sources/eomt_zeroshot_val_preds_coco.json" \
  --ckpt "${CKPT}" \
  --wandb \
  --wandb-group "${WANDB_RUN_GROUP}" \
  --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/eomt_mask2box_${SLURM_JOB_ID}.log" || {
  python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
    --gt-coco "${GT}" \
    --pred-coco "${OUT}/sources/eomt_zeroshot_val_preds_coco.json" \
    --out "${OUT}/sources/eomt_zeroshot_val_preds_coco.json" \
    --wandb --wandb-group "${WANDB_RUN_GROUP}" --wandb-name "${WANDB_NAME}_eval_fallback" \
    2>&1 | tee -a "${OUT}/sources/eomt_mask2box_${SLURM_JOB_ID}.log"
}

echo "Ablation 05 EoMT mask→box done."
