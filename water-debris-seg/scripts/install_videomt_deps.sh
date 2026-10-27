#!/usr/bin/env bash
# Install VidEoMT stack into the shared EoMT venv on scratch (Detectron2 + panopticapi + videomt reqs).
set -euo pipefail
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"
VENV="${HOME}/envs/wdseg-eomt"
VID="${SCRATCH}/third_party/videomt"

if [[ ! -d "${VENV}/bin" ]]; then
  echo "Missing ${VENV}; run setup_eomt_venv.sbatch first."
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -c "import torch; print('torch', torch.__version__)"

if ! python -c "import detectron2" 2>/dev/null; then
  pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
fi

pip install -q 'git+https://github.com/cocodataset/panopticapi.git' || true
# NumPy 2.x needs pycocotools >= 2.0.10 (NPY_OWNDATA); keep numpy aligned with torch/scipy stack.
pip install -q 'numpy==2.2.6' 'pycocotools>=2.0.10'
pip install -r "${VID}/requirements.txt"

python "${HOME}/projects/water-debris-seg-benchmark/water-debris-seg/scripts/videomt_env_smoke.py"
echo "VidEoMT deps OK in ${VENV}"
