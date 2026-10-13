#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab04-ft
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation04_eomt_flow_larsft-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
FLOW="${SCRATCH}/data/flow"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box"
GT="${OUT}/sources/flow_test_gt_coco.json"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/epoch=11-step=7812.ckpt"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation04_eomt_flow_larsft}"
export WANDB_NAME="${WANDB_NAME:-eomt_lars_panoptic_flow_${SLURM_JOB_ID}}"

python "${WDS}/scripts/verify_flow.py"
[[ -f "${GT}" ]] || { echo "Missing ${GT}"; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing ab03 ckpt"; exit 1; }

TEST="${FLOW}/test"
[[ -d "${FLOW}/FloW-Img/test" ]] && TEST="${FLOW}/FloW-Img/test"
IMAGES="${TEST}/images"
[[ -d "${TEST}/JPEGImages" ]] && IMAGES="${TEST}/JPEGImages"

python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
  --images "${IMAGES}" \
  --gt-coco "${GT}" \
  --out "${OUT}/sources/eomt_lars_panoptic_flow_preds_coco.json" \
  --lightning-ckpt "${CKPT}" \
  --num-classes 11 \
  --stuff-classes 0,1,2 \
  --wandb --wandb-group "${WANDB_RUN_GROUP}" --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/eomt_larsft_flow_${SLURM_JOB_ID}.log"

echo "Ablation 04 FloW LaRS-FT done."
