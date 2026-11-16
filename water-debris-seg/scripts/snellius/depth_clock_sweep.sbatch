#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J wdseg-depth-clock
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_clock_sweep-%j.out

# Depth benchmark D01, gate G1 clock check (README § G1 IMU clock check; G1 pilot report): camera-time offset sweep.
# Same 40 frames as the pilot diagnostics; Uhrig version only (the stereo check is the alignment measure).
#   OFFSETS (s, default -0.30..+0.40 step 0.05) x NSCANS (11 = Uhrig accumulation, 1 = single de-skewed scan)
# Outputs: /scratch-shared/$USER/water-debris-seg/depth_clock_sweep/<seq>/n<NSCANS>/off_<offset>/
set -euo pipefail
SEQ="${SEQ:-pohang00}"
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
RAW="${SCRATCH}/depth_raw/${SEQ}"
OUT="${SCRATCH}/depth_clock_sweep/${SEQ}"
WM="${REPO}/eval_outputs/water_debris_seg/ablations/11_yolo26_family/runs/runs/segment/water-debris-seg/lars11_yolo26m-seg/weights/best.pt"
B="${WDS}/scripts/depth/build_depth_gt.py"
OFFSETS="${OFFSETS:--0.30 -0.25 -0.20 -0.15 -0.10 -0.05 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40}"
mkdir -p "${SCRATCH}/logs" "${OUT}"
test -f "${RAW}/manifest.json" || { echo "download not finished: ${RAW}/manifest.json missing"; exit 1; }

# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
cd "${SCRATCH}"
python -I "${WDS}/scripts/depth/test_build_depth_gt.py"
FR=$(python -I -c "print(','.join(str(i) for i in range(250, 22183, 550)))")
for ns in 11 1; do
  for off in ${OFFSETS}; do
    python -I "$B" pohang --raw-dir "${RAW}" --seq "${SEQ}" --frames "${FR}" --out "${OUT}/n${ns}/off_${off}" \
      --water-model "${WM}" --device 0 --workers 16 --versions uhrig --n-scans "${ns}" --cam-time-offset-s "${off}"
    echo "done n${ns} off ${off}"
  done
done
