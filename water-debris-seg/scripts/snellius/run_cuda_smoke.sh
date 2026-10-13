#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=load_pytorch_module.sh
source "$ROOT/scripts/snellius/load_pytorch_module.sh"

srun --partition=gpu_a100 --gres=gpu:1 --time=00:10:00 --mem=16G \
  bash -lc "$(cat <<'INNER'
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
python3 - <<PY
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("alloc_ok", float(y[0, 0]))
    print("gpu0", torch.cuda.get_device_name(0))
PY
INNER
)"
