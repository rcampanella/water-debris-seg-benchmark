#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab11-cansurf-paper
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation11_cansurf_paper-%j.out

# Ablation 11: CANSURF val with the CANSURF paper's detection protocol (P / R / F1 over IoU 0.50:0.05:0.95 at conf
# > 0.25; cansurf_paper_metrics.py) for the five YOLO26 CANSURF best.pt, each checked against the paper's own script
# (verify_cansurf_paper_metrics.py). Separate from the plotted pycocotools AP50. Output: <ablation>/sources.
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
ABL="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family"
J="${SLURM_JOB_ID}"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
cd "${WDS}/scripts"
for M in ${MODELS:-n s m l x}; do
  W="${ABL}/runs/cansurf_yolo26${M}/weights/best.pt"
  python cansurf_paper_metrics.py --method "ab11_yolo26${M}" --weights "${W}" --out-dir "${ABL}/sources"
  python verify_cansurf_paper_metrics.py --weights "${W}" --ours "${ABL}/sources/cansurf_paper_ab11_yolo26${M}_${J}.json"
done
echo "ablation11 cansurf_paper done"
