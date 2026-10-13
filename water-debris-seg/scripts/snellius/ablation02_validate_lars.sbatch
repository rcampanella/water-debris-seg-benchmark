#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab02-val
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation02_validate-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATA_ZIPS="${SCRATCH}/data/lars/eomt_semantic_zips"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/mk2w7dr8/checkpoints/epoch=30-step=2356.ckpt"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/02_finetune_lars_semantic"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation02_finetune_lars_semantic}"
export WANDB_NAME="${WANDB_NAME:-eomt_lars_semantic_val_${SLURM_JOB_ID}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing fine-tune ckpt: ${CKPT}"
  exit 1
fi

python "${WDS}/scripts/pack_lars_semantic_zips.py" --full --skip-if-exists

cd "$EOMT"
python3 main.py validate \
  --config "${WDS}/configs/ablation02_lars_semantic_validate.yaml" \
  --trainer.devices 1 \
  --data.path "${DATA_ZIPS}" \
  --data.batch_size 4 \
  --ckpt_path "${CKPT}" \
  --trainer.logger.init_args.project "${WANDB_PROJECT}" \
  --trainer.logger.init_args.group "${WANDB_RUN_GROUP}" \
  --trainer.logger.init_args.name "${WANDB_NAME}" \
  --trainer.logger.init_args.save_dir "${WANDB_DIR}" \
  2>&1 | tee "${OUT}/sources/validate_${SLURM_JOB_ID}.log"

echo "Ablation 02 validate done."
