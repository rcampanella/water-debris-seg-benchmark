#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab08-yolo
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation08_yolo_cansurf-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
DATA="${HOME}/projects/water-debris-seg-benchmark/data/water_debris/cansurf_home"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/08_specialized_yolo"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation08_yolo_cansurf}"
export WANDB_NAME="${WANDB_NAME:-yolo11n_cansurf_${SLURM_JOB_ID}}"
export WANDB_PROJECT="${WANDB_PROJECT:-water-debris-seg}"
export YOLO_CONFIG_DIR="${SCRATCH}/artifacts/ultralytics_config"
mkdir -p "${YOLO_CONFIG_DIR}"

python "${WDS}/scripts/verify_cansurf.py"
pip install -q ultralytics
yolo settings wandb=True 2>/dev/null || true

yolo train \
  task=detect \
  data="${DATA}/data.yaml" \
  model=yolo11n.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project="${OUT}" \
  name=cansurf_train \
  exist_ok=True \
  2>&1 | tee "${OUT}/sources/train_${SLURM_JOB_ID}.log"

yolo val \
  task=detect \
  model="${OUT}/cansurf_train/weights/best.pt" \
  data="${DATA}/data.yaml" \
  imgsz=640 \
  device=0 \
  2>&1 | tee "${OUT}/sources/val_${SLURM_JOB_ID}.log"

echo "Ablation 08 YOLO done."
