#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-ab07-vmt-eval
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_eval-%j.out

# Eval-only VidEoMT on lars_ytvis_val (per-keyframe VIS AP via keyframe_patch).
# Usage: sbatch ablation07_videomt_eval.sbatch <model.pth> [MAX_NUM]
set -euo pipefail
CKPT="${1:?checkpoint path}"
MAX_NUM="${2:-50}"
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VID="${SCRATCH}/third_party/videomt"
WDS="${HOME}/projects/water-debris-seg-benchmark/water-debris-seg"
OUT="${WDS}/../eval_outputs/water_debris_seg/ablations/07_videomt_temporal/sources"
RUN_DIR="${SCRATCH}/artifacts/videomt/eval_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}/logs" "${OUT}" "${RUN_DIR}"
source "${SCRATCH}/envs/eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${VID}:${WDS}/scripts:${WDS}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation07_videomt_temporal}"
export WANDB_NAME="${WANDB_NAME:-videomt_lars_eval_${SLURM_JOB_ID}}"
cd "${VID}"
python "${WDS}/scripts/run_videomt_train_lars.py" --num-gpus 1 --eval-only \
  --config-file "${WDS}/configs/ablation07_videomt_lars_vitb.yaml" \
  MODEL.WEIGHTS "${CKPT}" MODEL.BACKBONE.TEST.MAX_NUM "${MAX_NUM}" OUTPUT_DIR "${RUN_DIR}" \
  2>&1 | tee "${OUT}/videomt_eval_${SLURM_JOB_ID}.log"
python "${WDS}/scripts/rescore_videomt_ytvis_keyframes.py" \
  --results "${RUN_DIR}/inference/results.json" \
  --out "${OUT}/videomt_rescore_keyframe_${SLURM_JOB_ID}.json"
