#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-eomt-smoke
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/smoke_eomt_lars-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
OUT="${REPO}/eval_outputs/water_debris_seg/smoke/eomt"
LARS_ZIPS="${SCRATCH}/data/lars/eomt_smoke_zips"
mkdir -p "${SCRATCH}/logs" "$OUT"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing venv ${VENV}; run setup_eomt_venv.sbatch first"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${REPO}/water-debris-seg/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${REPO}/water-debris-seg:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-smoke_eomt}"
export WANDB_NAME="${WANDB_NAME:-eomt_smoke_lars_${SLURM_JOB_ID}}"

python "${REPO}/water-debris-seg/scripts/verify_lars.py"
python "${REPO}/water-debris-seg/scripts/pack_lars_eomt_zips.py" --smoke 8

cd "$EOMT"
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_base_640.yaml \
  --trainer.devices 1 \
  --data.batch_size 2 \
  --data.path "${LARS_ZIPS}" \
  --trainer.limit_val_batches 2 \
  --trainer.logger.init_args.project "${WANDB_PROJECT}" \
  --trainer.logger.init_args.group "${WANDB_RUN_GROUP}" \
  --trainer.logger.init_args.name "${WANDB_NAME}" \
  --trainer.logger.init_args.save_dir "${WANDB_DIR}" \
  2>&1 | tee "${OUT}/validate_log.txt"

echo "EoMT LaRS smoke validate log -> ${OUT}/validate_log.txt"
