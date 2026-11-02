#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH -J wdseg-ab11-latency
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation11_latency-%j.out

# Ablation 11: time every trained YOLO26 / YOLO26-seg best.pt in one job on one GPU (bench_yolo26_latency.py: PyTorch
# FP16 and TensorRT FP16 engines). --exclusive keeps other jobs off the node while timing. Env: ~/envs/wdseg-trt
# (wdseg-eomt's torch / ultralytics + TensorRT; envs/trt_latency_venv.txt).
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
ABL="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-trt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "${WDS}/scripts/bench_yolo26_latency.py" 2>&1 | tee "${ABL}/sources/ab11_latency_${SLURM_JOB_ID}.log"
