#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -J wdseg-depth-zs
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_eval_zeroshot-%j.out

# Depth benchmark D01, step 5: zero-shot scoring of the released YOLO26-depth n-x (shipped calibration) on one source,
# day and night test splits (eval_depth.py: MDEC relative + KITTI absolute, bands; README D12, D13).
#   SOURCE=pohang|multiaqua
set -euo pipefail
SOURCE="${SOURCE:?set SOURCE=pohang|multiaqua}"
REPO="${HOME}/projects/water-debris-seg-benchmark"
DATA="${REPO}/data/water_debris/depth_d01/${SOURCE}"
W="${REPO}/data/water_debris/checkpoints/yolo/depth"
OUT="${REPO}/eval_outputs/water_debris_depth/ablations/D01_rgb/sources/zeroshot"
mkdir -p "${OUT}"
if [[ "${SOURCE}" == pohang ]]; then K="1814.506142073043,1814.506142073043,1026.197713069134,524.7984412358129"
else K="1365.43274,1365.43274,1094.44934,624.207886"; fi
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
echo "scorer sha256: $(sha256sum "${REPO}/water-debris-seg/scripts/depth/eval_depth.py" | cut -d' ' -f1)"
echo "mdec commit: $(git -C "${REPO}/third_party/mdec_benchmark" rev-parse HEAD); patch: $(git -C "${REPO}/third_party/mdec_benchmark" diff | sha256sum | cut -d' ' -f1)"
cd /scratch-shared/${USER}/water-debris-seg
for s in n s m l x; do
  for split in test_day test_night; do
    python "${REPO}/water-debris-seg/scripts/depth/eval_depth.py" --data "${DATA}" --split "${split}" \
      --weights "${W}/yolo26${s}-depth.pt" --arm zero-shot --K "${K}" --device 0 \
      --out "${OUT}/zs_${SOURCE}_yolo26${s}_${split}_${SLURM_JOB_ID}"
  done
done
echo "done zeroshot ${SOURCE}"
