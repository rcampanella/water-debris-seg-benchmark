#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH -J wdseg-lars-official
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/lars_official-%j.out

# LaRS val with the official LaRS evaluator (lojzezust/lars_evaluator), separate from our PQ (RESULTS.md section
# "LaRS official evaluator"). Re-scores the checkpoint with our scorer while exporting its panoptic maps as LaRS PNGs,
# then runs the official evaluator on them.
#   KIND=fix1 (EoMT: MODEL=lars_panoptic|coco133, CKPT, MAPPING)   KIND=sam (SOURCE=eomt|yolo, WEIGHTS, THRESHOLDS)
#   METHOD (name of the prediction dir and of the output JSON)
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
EOMT="${REPO}/data/water_debris/third_party/eomt"
OUT="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources"
KIND="${KIND:?set KIND=fix1|sam}"; METHOD="${METHOD:?set METHOD}"
PRED="${REPO}/data/water_debris/lars_eval/preds/${METHOD}"
SCORER_JSON="${OUT}/lars_official_scorer_${METHOD}_${SLURM_JOB_ID}.json"
mkdir -p "/scratch-shared/${USER}/water-debris-seg/logs" "${OUT}"
rm -rf "${PRED}"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
if [[ "${KIND}" == "fix1" ]]; then
  python "${WDS}/scripts/eval_eomt_fairness.py" --model "${MODEL:?}" --ckpt "${CKPT:?}" --dataset lars \
    --mapping "${MAPPING:-tightened}" --export-png "${PRED}" --out "${SCORER_JSON}"
else
  # shellcheck disable=SC2086
  python "${WDS}/scripts/eval_sam_panoptic.py" --source "${SOURCE:?}" --weights "${WEIGHTS:?}" --split val \
    --thresholds ${THRESHOLDS:-0.25} --export-png "${PRED}" --out "${SCORER_JSON}"
fi
deactivate

"${HOME}/envs/lars-eval/bin/python" "${WDS}/scripts/run_lars_evaluator.py" --method "${METHOD}" --scorer-json "${SCORER_JSON}"
echo "lars_official ${METHOD} done"
