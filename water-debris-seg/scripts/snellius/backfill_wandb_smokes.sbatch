#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-wandb-smoke
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/backfill_wandb_smokes-%j.out

# Re-log Wave-2 smokes to project water-debris-seg (YOLO then EoMT micro-validate).
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
EOMT="${SCRATCH}/third_party/eomt"
mkdir -p "${SCRATCH}/logs"

# shellcheck disable=SC1091
source "${SCRATCH}/envs/eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}:${EOMT}:${PYTHONPATH:-}"

export WANDB_RUN_GROUP=smoke_yolo
export WANDB_NAME="yolo11n_cansurf_smoke_${SLURM_JOB_ID}"
bash "${WDS}/scripts/snellius/smoke_yolo_cansurf.sbatch" 2>/dev/null || {
  DATA="${SCRATCH}/data/cansurf"
  OUT="${REPO}/eval_outputs/water_debris_seg/smoke/yolo"
  pip install -q ultralytics
  yolo settings wandb=True 2>/dev/null || true
  yolo train task=detect data="${DATA}/data.yaml" model=yolo11n.pt epochs=1 imgsz=640 batch=8 device=0 \
    fraction=0.002 project="${OUT}" name=microrun_wandb exist_ok=True
}

export WANDB_RUN_GROUP=smoke_eomt
export WANDB_NAME="eomt_smoke_lars_${SLURM_JOB_ID}"
python "${WDS}/scripts/pack_lars_eomt_zips.py" --smoke 8
cd "$EOMT"
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_base_640.yaml \
  --trainer.devices 1 \
  --data.batch_size 2 \
  --data.path "${SCRATCH}/data/lars/eomt_smoke_zips" \
  --trainer.limit_val_batches 2 \
  --trainer.logger.init_args.project "${WANDB_PROJECT}" \
  --trainer.logger.init_args.group "${WANDB_RUN_GROUP}" \
  --trainer.logger.init_args.name "${WANDB_NAME}" \
  --trainer.logger.init_args.save_dir "${WANDB_DIR}"

echo "Smoke wandb backfill done."
