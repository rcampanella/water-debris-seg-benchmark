#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab01-eomt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation01_eomt_lars-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATA_ZIPS="${SCRATCH}/data/lars/eomt_full_zips"
CKPT="${SCRATCH}/artifacts/checkpoints/eomt/coco_panoptic_eomt_base_640_2x.bin"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/01_zeroshot_lars"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"

export PYTHONPATH="${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation01_zeroshot_lars}"
export WANDB_NAME="${WANDB_NAME:-eomt_base640_2x_lars_val_${SLURM_JOB_ID}}"

bash "${WDS}/scripts/download_eomt_ckpt.sh"
python "${WDS}/scripts/pack_lars_eomt_zips.py" --full
python "${WDS}/scripts/verify_lars.py"

cd "$EOMT"
python3 main.py validate \
  --config "${WDS}/configs/ablation01_lars_zeroshot.yaml" \
  --trainer.devices 1 \
  --data.path "${DATA_ZIPS}" \
  --data.batch_size 4 \
  --model.init_args.ckpt_path "${CKPT}" \
  --trainer.logger.init_args.project "${WANDB_PROJECT}" \
  --trainer.logger.init_args.group "${WANDB_RUN_GROUP}" \
  --trainer.logger.init_args.name "${WANDB_NAME}" \
  --trainer.logger.init_args.save_dir "${WANDB_DIR}" \
  2>&1 | tee "${OUT}/sources/validate_${SLURM_JOB_ID}.log"

echo "Ablation 01 validate done. Log: ${OUT}/sources/validate_${SLURM_JOB_ID}.log"
