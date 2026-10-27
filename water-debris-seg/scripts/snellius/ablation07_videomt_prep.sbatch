#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -J wdseg-ab07-prep
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_prep-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
VID="${SCRATCH}/third_party/videomt"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/verify_lars_sequences.py" | tee "${OUT}/sources/sequences_check_${SLURM_JOB_ID}.log"
test -d "${VID}" || { echo "Clone videomt: water-debris-seg/scripts/clone_third_party.sh"; exit 1; }
echo "VidEoMT repo present at ${VID}. Full train/eval wiring is next (ablation 07)."
echo "Use EoMT ablation 03 ckpt as still-frame baseline for comparison."
