#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH -J wdseg-qual-masks
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/qual_dump_masks-%j.out

# Qualitative only: saves EoMT masks for the images in qualitative/selection.json and for
# LaRS val, using the benchmarked inference code and registry configs. No metrics.
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
python "${WDS}/scripts/qual_dump_masks.py" "${MODE:-all}"
