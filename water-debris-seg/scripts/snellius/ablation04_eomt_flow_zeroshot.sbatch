#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH -J wdseg-ab04-zs
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation04_eomt_flow_zeroshot-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
FLOW="${SCRATCH}/data/flow"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box"
GT="${OUT}/sources/flow_test_gt_coco.json"
CKPT="${SCRATCH}/artifacts/checkpoints/eomt/coco_panoptic_eomt_base_640_2x.bin"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation04_eomt_flow_zeroshot}"
export WANDB_NAME="${WANDB_NAME:-eomt_flow_mask2box_${SLURM_JOB_ID}}"

python "${WDS}/scripts/verify_flow.py"

TEST="${FLOW}/test"
[[ -d "${FLOW}/FloW-Img/test" ]] && TEST="${FLOW}/FloW-Img/test"
IMAGES="${TEST}/images"
[[ -d "${TEST}/JPEGImages" ]] && IMAGES="${TEST}/JPEGImages"
LABELS="${TEST}/labels"
ANNS="${TEST}/Annotations"
[[ -d "${TEST}/annotations" ]] && ANNS="${TEST}/annotations"

if [[ ! -f "${GT}" ]]; then
  if [[ -d "${LABELS}" ]]; then
    python "${WDS}/scripts/masks_to_coco_boxes.py" yolo-split \
      --images "${IMAGES}" --labels "${LABELS}" --out "${GT}"
  else
    python "${WDS}/scripts/masks_to_coco_boxes.py" voc-split \
      --images "${IMAGES}" --annotations "${ANNS}" --out "${GT}"
  fi
fi

bash "${WDS}/scripts/download_eomt_ckpt.sh"

python "${WDS}/scripts/eval_eomt_cansurf_mask2box.py" \
  --images "${IMAGES}" \
  --gt-coco "${GT}" \
  --out "${OUT}/sources/eomt_zeroshot_flow_preds_coco.json" \
  --ckpt "${CKPT}" \
  --wandb --wandb-group "${WANDB_RUN_GROUP}" --wandb-name "${WANDB_NAME}" \
  2>&1 | tee "${OUT}/sources/eomt_zeroshot_flow_${SLURM_JOB_ID}.log"

echo "Ablation 04 FloW zero-shot done."
