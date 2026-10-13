#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH -J wdseg-ab08-wb
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation08_yolo_wandb_val-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATA="${SCRATCH}/data/cansurf"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/08_specialized_yolo"
WEIGHTS="${OUT}/cansurf_train/weights/best.pt"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation08_yolo_cansurf}"
export WANDB_NAME="${WANDB_NAME:-yolo11n_cansurf_val_wandb_${SLURM_JOB_ID}}"

python "${WDS}/scripts/verify_cansurf.py"
pip install -q ultralytics
yolo settings wandb=True 2>/dev/null || true

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Missing weights ${WEIGHTS}; run ablation08_yolo_cansurf.sbatch first."
  exit 1
fi

python "${WDS}/scripts/log_yolo_val_wandb.py" \
  --model "${WEIGHTS}" \
  --data "${DATA}/data.yaml" \
  --group "${WANDB_RUN_GROUP}" \
  --name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/wandb_val_${SLURM_JOB_ID}.log"

echo "Ablation 08 W&B val log done."
