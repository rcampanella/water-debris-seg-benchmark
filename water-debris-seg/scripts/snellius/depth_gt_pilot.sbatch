#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J wdseg-depth-gt
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_gt_pilot-%j.out

# Depth benchmark D01, step 2 (README § Ground truth, Launch plan): PILOT ground-truth build for gate G1.
#   SOURCE=pohang00  : every 50th stereo frame, Uhrig + raw versions (build_depth_gt.py pohang), then diagnostics on
#                      40 frames: camera-time offsets -0.1/-0.05/+0.05/+0.1 s, SLAM rotation instead of LiDAR-IMU,
#                      no gyro-bias correction, SGM-invalid points kept (stats only, separate output folders)
#   SOURCE=multiaqua : 50 evenly spaced frames per sequence (adr1, bl1, lj1-lj4), raw version
# Outputs: /scratch-shared/$USER/water-debris-seg/depth_gt_pilot/<source>/ ; summary + renders by the 'summarize'
# sub-command afterwards (light, login node).
set -euo pipefail
SOURCE="${SOURCE:?set SOURCE=pohang00|multiaqua}"
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${SCRATCH}/depth_gt_pilot"
DIAG="${SCRATCH}/depth_gt_pilot_diag"
WM="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family/runs/runs/segment/water-debris-seg/lars11_yolo26m-seg/weights/best.pt"
B="${WDS}/scripts/depth/build_depth_gt.py"
NW="${WORKERS:-16}"
mkdir -p "${SCRATCH}/logs" "${OUT}" "${DIAG}"

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "water model sha256: $(sha256sum "${WM}" | cut -d' ' -f1)"
cd "${SCRATCH}"   # never run from inside a data directory

python -I "${WDS}/scripts/depth/test_build_depth_gt.py"

if [[ "${SOURCE}" == pohang0[0-5] ]]; then
  RAW="${SCRATCH}/depth_raw/${SOURCE}"
  test -f "${RAW}/manifest.json" || { echo "download not finished: ${RAW}/manifest.json missing"; exit 1; }
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames every:50 --out "${OUT}" \
    --water-model "${WM}" --device 0 --workers "${NW}"
  # diagnostics (stats only matter): 40 frames spread over the sequence
  FR=$(python -I -c "print(','.join(str(i) for i in range(250, 22183, 550)))")
  for off in -0.1 -0.05 0.05 0.1; do
    python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames "${FR}" --out "${DIAG}/camoff_${off}" \
      --water-model "${WM}" --device 0 --workers "${NW}" --versions uhrig --cam-time-offset-s "${off}"
  done
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames "${FR}" --out "${DIAG}/rot_slam" \
    --water-model "${WM}" --device 0 --workers "${NW}" --versions uhrig --rotation-source slam
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames "${FR}" --out "${DIAG}/no_gyro_bias" \
    --water-model "${WM}" --device 0 --workers "${NW}" --versions uhrig --gyro-bias-window-s 0
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames "${FR}" --out "${DIAG}/sgm_invalid_keep" \
    --water-model "${WM}" --device 0 --workers "${NW}" --versions uhrig --sgm-invalid keep
  python -I "$B" pohang --raw-dir "${RAW}" --seq "${SOURCE}" --frames "${FR}" --out "${DIAG}/camoff_0" \
    --water-model "${WM}" --device 0 --workers "${NW}" --versions uhrig
elif [[ "${SOURCE}" == multiaqua ]]; then
  python -I "$B" multiaqua --zip "${SCRATCH}/depth_raw/multiaqua/MULTIAQUA_full.zip" --frames per_seq:50 \
    --out "${OUT}" --water-model "${WM}" --device 0 --workers "${NW}"
else
  echo "unknown SOURCE ${SOURCE}"; exit 1
fi
echo "done ${SOURCE}"
