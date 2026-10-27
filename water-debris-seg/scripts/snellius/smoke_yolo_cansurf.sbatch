#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -J wdseg-yolo-smoke
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/smoke_yolo_cansurf-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
OUT="${REPO}/eval_outputs/water_debris_seg/smoke/yolo"
DATA="${HOME}/projects/water-debris-seg-benchmark/data/water_debris/cansurf_home"
mkdir -p "${SCRATCH}/logs" "$OUT"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing venv; run setup_eomt_venv.sbatch first"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${REPO}/water-debris-seg/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${REPO}/water-debris-seg:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-smoke_yolo}"
export WANDB_NAME="${WANDB_NAME:-yolo11n_cansurf_micro_${SLURM_JOB_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"

python "${REPO}/water-debris-seg/scripts/verify_cansurf.py"
pip install -q ultralytics
yolo settings wandb=True 2>/dev/null || true

yolo train \
  task=detect \
  data="${DATA}/data.yaml" \
  model=yolo11n.pt \
  epochs=1 \
  imgsz=640 \
  batch=8 \
  device=0 \
  fraction=0.002 \
  project="${OUT}" \
  name=microrun \
  exist_ok=True \
  2>&1 | tee "${OUT}/train_log.txt"

echo "YOLO smoke done -> ${OUT}/microrun"
