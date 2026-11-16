#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -J wdseg-depth-gtbuild
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_gt_build-%j.out

# Depth benchmark D01, step 2: FULL ground-truth build, unadapted (README D11, 2026-11-13): no water filter
# (points on the water mask are kept and tagged), released camera timestamps, undistorted left image.
#   SOURCE=pohang00..05 : every stereo frame, VERSIONS (default raw; uhrig later)
#   SOURCE=multiaqua    : every frame with LiDAR, raw
# Output: /scratch-shared/$USER/water-debris-seg/depth_gt/unadapted_v1/<source>/
set -euo pipefail
SOURCE="${SOURCE:?set SOURCE=pohang00..05|multiaqua}"
VERSIONS="${VERSIONS:-raw}"
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${SCRATCH}/depth_gt/unadapted_v1"
WM="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family/runs/runs/segment/water-debris-seg/lars11_yolo26m-seg/weights/best.pt"
B="${WDS}/scripts/depth/build_depth_gt.py"
mkdir -p "${SCRATCH}/logs" "${OUT}"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
echo "water model sha256: $(sha256sum "${WM}" | cut -d' ' -f1)"
echo "builder sha256: $(sha256sum "${B}" | cut -d' ' -f1)"
cd "${SCRATCH}"
python -I "${WDS}/scripts/depth/test_build_depth_gt.py"

if [[ "${SOURCE}" == pohang0[0-5] ]]; then
  RAW="${SCRATCH}/depth_raw/${SOURCE}"
  test -f "${RAW}/manifest.json" || { echo "download not finished: ${RAW}/manifest.json missing"; exit 1; }
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames every:1 --versions "${VERSIONS}" \
    --out "${OUT}" --water-model "${WM}" --water-filter off --device 0 --workers 16 --cache-every 50
elif [[ "${SOURCE}" == multiaqua ]]; then
  python -I "$B" multiaqua --zip "${SCRATCH}/depth_raw/multiaqua/MULTIAQUA_full.zip" --frames per_seq:100000 \
    --out "${OUT}" --water-model "${WM}" --water-filter off --device 0 --workers 16 --cache-every 50
else
  echo "unknown SOURCE ${SOURCE}"; exit 1
fi
echo "done ${SOURCE}"
