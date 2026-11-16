#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH -J wdseg-depth-latency
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_latency-%j.out

# D01 depth latency: every D01 fine-tuned YOLO26-depth best.pt + the 5 released weights in one job on one GPU
# (depth/bench_depth_latency.py: PyTorch FP16 and TensorRT FP16 engines), as ablation 11. --exclusive keeps other jobs off the node while timing. Env: ~/envs/wdseg-trt
# (wdseg-eomt's torch / ultralytics + TensorRT; envs/trt_latency_venv.txt).
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_depth/ablations/D01_rgb/sources/latency"
mkdir -p "${OUT}"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-trt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "${WDS}/scripts/depth/bench_depth_latency.py" "$@" 2>&1 | tee "${OUT}/d01_latency_${SLURM_JOB_ID}.log"
