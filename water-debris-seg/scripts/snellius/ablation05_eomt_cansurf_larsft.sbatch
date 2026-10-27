#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab05-larsft
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation05_eomt_cansurf_larsft-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/05_cansurf_small_object"
GT="${OUT}/sources/cansurf_val_gt_coco.json"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/epoch=11-step=7812.ckpt"
PRED="${OUT}/sources/eomt_lars_panoptic_ft_val_preds_coco.json"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation05_eomt_cansurf_larsft}"
export WANDB_NAME="${WANDB_NAME:-eomt_lars_panoptic_cansurf_${SLURM_JOB_ID}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing ab03 ckpt: ${CKPT}"
  exit 1
fi

python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
  --gt-coco "${GT}" \
  --out "${PRED}" \
  --lightning-ckpt "${CKPT}" \
  --num-classes 11 \
  --stuff-classes 0,1,2 \
  --wandb \
  --wandb-group "${WANDB_RUN_GROUP}" \
  --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/eomt_larsft_mask2box_${SLURM_JOB_ID}.log" || {
  python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
    --gt-coco "${GT}" \
    --pred-coco "${PRED}" \
    --out "${PRED}" \
    --wandb --wandb-group "${WANDB_RUN_GROUP}" --wandb-name "${WANDB_NAME}_eval_fallback" \
    2>&1 | tee -a "${OUT}/sources/eomt_larsft_mask2box_${SLURM_JOB_ID}.log"
}

echo "Ablation 05 LaRS panoptic FT mask→box done."
