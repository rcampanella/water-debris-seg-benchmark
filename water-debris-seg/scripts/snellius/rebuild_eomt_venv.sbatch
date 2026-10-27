#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-venv-rebuild
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/rebuild_eomt_venv-%j.out

# Rebuild the scratch venv after the 2026-10-23 scratch purge deleted package files.
# Versions come from water-debris-seg/envs/eomt_venv_recovered_2026-10-23.txt (recovered from the old venv's
# *.dist-info folder names). The damaged venv is renamed, not deleted. VidEoMT-only deps (detectron2,
# panopticapi; git builds) are skipped: no VidEoMT runs are planned. torch/torchvision come from the CUDA 12.6 index.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${VENV:-${SCRATCH}/envs/eomt}"   # 2026-10-23: rebuilt on home (${HOME}/envs/wdseg-eomt), scratch copies were purged again
REQ="${HOME}/projects/water-debris-seg-benchmark/water-debris-seg/envs/eomt_venv_recovered_2026-10-23.txt"
mkdir -p "${SCRATCH}/logs"

if [[ -d "${VENV}" && ! -e "${VENV}/.rebuilt-2026-10-23" ]]; then
  mv "${VENV}" "${VENV}_purged_$(date +%F)"
fi
mkdir -p "$(dirname "${VENV}")"

module load 2025
module load Python/3.13.5-GCCcore-14.3.0
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
# everything else at the recovered versions, minus torch-bundled and VidEoMT git-built packages
grep -v -E '^#|^(torch|torchvision|triton|nvidia_[a-z0-9_]+|pip|setuptools|detectron2|panopticapi)==' "${REQ}" \
  | sed -E 's/^([A-Za-z0-9_.]+)==/\1==/' > "${SCRATCH}/logs/rebuild_req_${SLURM_JOB_ID}.txt"
pip install -r "${SCRATCH}/logs/rebuild_req_${SLURM_JOB_ID}.txt"

python - <<'PY'
import importlib
for m in ["torch", "torchvision", "numpy", "PIL", "lightning", "timm", "pycocotools.cocoeval", "ultralytics", "wandb",
          "torchmetrics", "scipy", "cv2"]:
    mod = importlib.import_module(m)
    print(f"{m:22s} {getattr(mod, '__version__', 'ok')}")
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
touch "${VENV}/.rebuilt-2026-10-23"
echo "venv rebuilt: ${VENV}"
