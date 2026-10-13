#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH -J wdseg-ab04-gt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation04_flow_gt-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
FLOW="${SCRATCH}/data/flow"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/04_flow_mask2box"
GT="${OUT}/sources/flow_test_gt_coco.json"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${SCRATCH}/envs/eomt/bin/activate" 2>/dev/null || true
export PYTHONPATH="${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/verify_flow.py"

# Resolve test split paths (see docs/FLOW_DATA.md)
TEST="${FLOW}/test"
[[ -d "${FLOW}/FloW-Img/test" ]] && TEST="${FLOW}/FloW-Img/test"
IMAGES="${TEST}/images"
[[ -d "${TEST}/JPEGImages" ]] && IMAGES="${TEST}/JPEGImages"
LABELS="${TEST}/labels"
ANNS="${TEST}/Annotations"
[[ -d "${TEST}/annotations" ]] && ANNS="${TEST}/annotations"

if [[ -d "${LABELS}" ]]; then
  python "${WDS}/scripts/masks_to_coco_boxes.py" yolo-split \
    --images "${IMAGES}" --labels "${LABELS}" --out "${GT}"
else
  python "${WDS}/scripts/masks_to_coco_boxes.py" voc-split \
    --images "${IMAGES}" --annotations "${ANNS}" --out "${GT}"
fi
echo "FloW test GT -> ${GT}"
