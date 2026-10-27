#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -J wdseg-ab07-ytld
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_lars_ytvis_load_smoke-%j.out
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
VID="${SCRATCH}/third_party/videomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${VENV}/bin/activate"
export PYTHONPATH="${VID}:${WDS}/scripts:${WDS}:${PYTHONPATH:-}"

for split in val train; do
  python "${WDS}/scripts/lars_ytvis_load_smoke.py" --split "${split}" \
    2>&1 | tee "${OUT}/sources/lars_ytvis_load_smoke_${split}_${SLURM_JOB_ID}.log"
done

echo "LaRS YTVIS Detectron2 load smoke done."
