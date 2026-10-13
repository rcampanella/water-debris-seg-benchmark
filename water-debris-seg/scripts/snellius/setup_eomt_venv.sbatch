#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-eomt-venv
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/setup_eomt_venv-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
mkdir -p "${SCRATCH}/logs" "${SCRATCH}/envs"

module load 2025
module load Python/3.13.5-GCCcore-14.3.0

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

# EoMT upstream pins torch 2.7; install on GPU node (CUDA wheels).
pip install -r "${EOMT}/requirements.txt"

python - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

echo "EoMT venv ready: ${VENV}"
echo "activate: source ${VENV}/bin/activate"
