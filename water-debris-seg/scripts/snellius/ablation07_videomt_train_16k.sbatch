#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-ab07-vmt16k
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_train_16k-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
VID="${SCRATCH}/third_party/videomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
CKPT_DIR="${SCRATCH}/artifacts/checkpoints/videomt"
SEG="${CKPT_DIR}/yt_2019_base_segmenter.pth"
RUN_DIR="${SCRATCH}/artifacts/videomt/lars_vitb_16k_${SLURM_JOB_ID}"
CFG="${WDS}/configs/ablation07_videomt_lars_vitb_16k.yaml"
mkdir -p "${SCRATCH}/logs" "$OUT/sources" "${CKPT_DIR}" "${RUN_DIR}"

source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${VID}:${WDS}/scripts:${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation07_videomt_temporal}"
export WANDB_NAME="${WANDB_NAME:-videomt_lars_vitb_16k_${SLURM_JOB_ID}}"

test -f "${SCRATCH}/data/lars/ytvis/lars_ytvis_train.json"
test -f "${SEG}"

cd "${VID}"
python "${WDS}/scripts/run_videomt_train_lars.py" \
  --num-gpus 1 \
  --config-file "${CFG}" \
  OUTPUT_DIR "${RUN_DIR}" \
  2>&1 | tee "${OUT}/sources/videomt_train_16k_${SLURM_JOB_ID}.log"

echo "VidEoMT LaRS 16k train finished. OUTPUT_DIR=${RUN_DIR}"
