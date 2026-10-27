#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab01-predremap
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation01_lars_pq_predremap-%j.out

# Ablation 01 re-run (decision log 2026-10-13): LaRS val PQ in native LaRS space.
# Step 1 validates the script: ab03 native must reproduce EoMT validate val_pq_all 0.3945 (job 27660922).
# Step 2 scores frozen COCO-133 EoMT with prediction-side remap (configs/coco133_to_lars_pred_map.json).
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/sources"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
python "${WDS}/scripts/eval_eomt_lars_panoptic_pq.py" --mode native \
  --ckpt "${REPO}/data/water_debris/checkpoints/ab03_panoptic_67kk33hk/epoch=11-step=7812.ckpt" \
  --out "${OUT}/pq_native_ab03_check_${SLURM_JOB_ID}.json"
python "${WDS}/scripts/eval_eomt_lars_panoptic_pq.py" --mode coco133 \
  --ckpt "${SCRATCH}/artifacts/checkpoints/eomt/coco_panoptic_eomt_base_640_2x.bin" \
  --out "${OUT}/pq_zeroshot_predremap_${SLURM_JOB_ID}.json"
