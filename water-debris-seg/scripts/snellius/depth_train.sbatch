#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH -J wdseg-depth-ft
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_train-%j.out

# Depth benchmark D01, step 4: fine-tune one released YOLO26-depth model on one source, then score best / last
# (train_yolo26_depth.py). SOURCE=pohang|multiaqua MODEL=yolo26{n,s,m,l,x}-depth [EPOCHS FRACTION NAME for smoke tests]
set -euo pipefail
SOURCE="${SOURCE:?}"; MODEL="${MODEL:?}"
REPO="${HOME}/projects/water-debris-seg-benchmark"; WDS="${REPO}/water-debris-seg"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh" >/dev/null   # as ablation 11; prints nothing (API key never echoed)
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARGS=(--source "${SOURCE}" --model "${MODEL}")
[[ -n "${EPOCHS:-}" ]] && ARGS+=(--epochs "${EPOCHS}")
[[ -n "${FRACTION:-}" ]] && ARGS+=(--fraction "${FRACTION}")
[[ -n "${NAME:-}" ]] && ARGS+=(--name "${NAME}")
python "${WDS}/scripts/depth/train_yolo26_depth.py" "${ARGS[@]}"
