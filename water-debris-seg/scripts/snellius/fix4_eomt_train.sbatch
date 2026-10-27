#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH -J wdseg-fix4-eomt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/fix4_eomt_train-%j.out

# Fairness fix 4: EoMT full fine-tuning vs LoRA with the held-out protocol (FIX4_EOMT_LORA.md).
#   ARM=full|lora   STAGE=A|B   DATASET=lars|flow|cansurf (default lars; flow/cansurf = fix 3 on SAM masks)   EPOCHS (stage A budget, default 12)   LIMIT (smoke: fraction of batches)
#   TAG (variant name, e.g. e24 / lr5; run dir eomt_<dataset>_<arm>_<TAG>_stage<S>)   LR_MULT (base lr multiplier, default 1)
# Stage B reads the epoch count from its stage A CSV and, when done, scores the last checkpoint with fix 1's scorer.
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
FIX="${REPO}/eval_outputs/water_debris_seg/fairness_fixes"
ARM="${ARM:?set ARM=full|lora}"; STAGE="${STAGE:?set STAGE=A|B}"; DATASET="${DATASET:-lars}"
VARIANT="${ARM}${TAG:+_${TAG}}"
NAME="eomt_${DATASET}_${VARIANT}_stage${STAGE}"
mkdir -p "/scratch-shared/${USER}/water-debris-seg/logs" "${FIX}/sources"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
ARGS=(--dataset "${DATASET}" --arm "${ARM}" --stage "${STAGE}" --limit-batches "${LIMIT:-1.0}" --lr-mult "${LR_MULT:-1.0}" --tag "${TAG:-}")
if [[ "${STAGE}" == "A" ]]; then
  ARGS+=(--epochs "${EPOCHS:-12}")
else
  ARGS+=(--epochs-from "$(ls -d "${FIX}/runs/eomt_${DATASET}_${VARIANT}_stageA/csv"/version_*/metrics.csv | tail -1)")
fi
python "${WDS}/scripts/train_eomt_heldout.py" "${ARGS[@]}" 2>&1 | tee "${FIX}/sources/train_${NAME}_${SLURM_JOB_ID}.log"

if [[ "${STAGE}" == "B" ]]; then
  CK=$(find "${FIX}/runs/${NAME}" -name "*.ckpt" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)   # last checkpoint of this run
  [[ -n "${CK}" ]] || { echo "no checkpoint found under ${FIX}/runs/${NAME}"; exit 1; }
  EVAL_MODEL=lars_panoptic; [[ "${DATASET}" != "lars" ]] && EVAL_MODEL=box_instance
  MODEL="${EVAL_MODEL}" CKPT="${CK}" DATASET="${DATASET}" TAG="eomt_${VARIANT}" LIMIT="${LIMIT_EVAL:-0}" \
    bash "${WDS}/scripts/snellius/fix1_eomt_eval.sbatch"
fi
echo "${NAME} done"
