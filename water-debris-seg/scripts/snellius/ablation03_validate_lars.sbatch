#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab03-val
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation03_validate-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATA_ZIPS="${SCRATCH}/data/lars/eomt_full_zips"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/epoch=11-step=7812.ckpt"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/03_finetune_lars_panoptic"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation03_finetune_lars_panoptic}"
export WANDB_NAME="${WANDB_NAME:-eomt_lars_panoptic_val_${SLURM_JOB_ID}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing fine-tune ckpt: ${CKPT}"
  exit 1
fi

cd "$EOMT"
python3 main.py validate \
  --config "${WDS}/configs/ablation03_lars_panoptic_validate.yaml" \
  --trainer.devices 1 \
  --data.path "${DATA_ZIPS}" \
  --data.batch_size 4 \
  --ckpt_path "${CKPT}" \
  --trainer.logger.init_args.project "${WANDB_PROJECT}" \
  --trainer.logger.init_args.group "${WANDB_RUN_GROUP}" \
  --trainer.logger.init_args.name "${WANDB_NAME}" \
  --trainer.logger.init_args.save_dir "${WANDB_DIR}" \
  2>&1 | tee "${OUT}/sources/validate_${SLURM_JOB_ID}.log"

echo "Ablation 03 validate done."
