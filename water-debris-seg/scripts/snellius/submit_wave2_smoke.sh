#!/usr/bin/env bash
# Submit YOLO + EoMT smoke jobs after EoMT venv exists and passes CUDA import.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Waiting for venv at ${VENV}..."
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python - <<PY
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY

YOLO_JOB=$(sbatch "${REPO}/water-debris-seg/scripts/snellius/smoke_yolo_cansurf.sbatch" | awk '{print $4}')
EOMT_JOB=$(sbatch "${REPO}/water-debris-seg/scripts/snellius/smoke_eomt_lars.sbatch" | awk '{print $4}')
echo "submitted yolo_smoke=${YOLO_JOB} eomt_smoke=${EOMT_JOB}"
