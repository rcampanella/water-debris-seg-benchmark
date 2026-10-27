#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab07-ytvis
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_lars_ytvis_export-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
YTVIS="${SCRATCH}/data/lars/ytvis"
mkdir -p "${SCRATCH}/logs" "$OUT/sources" "$YTVIS"

source "${VENV}/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/export_lars_ytvis_json.py" \
  --split both \
  --out-dir "${YTVIS}" \
  2>&1 | tee "${OUT}/sources/lars_ytvis_export_${SLURM_JOB_ID}.log"

for split in train val; do
  cp -f "${YTVIS}/lars_ytvis_${split}.json" "${OUT}/sources/lars_ytvis_${split}.json"
done

echo "LaRS YTVIS export done under ${YTVIS}"
