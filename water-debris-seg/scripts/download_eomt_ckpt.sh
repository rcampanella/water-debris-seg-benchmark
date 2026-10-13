#!/usr/bin/env bash
set -euo pipefail
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"
DEST="${SCRATCH}/artifacts/checkpoints/eomt"
mkdir -p "$DEST/sources"

# EoMT-B COCO panoptic 640 2x (model zoo dinov2.md)
NAME="coco_panoptic_eomt_base_640_2x"
URL="https://huggingface.co/tue-mps/${NAME}/resolve/main/pytorch_model.bin"
OUT="${DEST}/${NAME}.bin"

if [[ -f "$OUT" ]]; then
  echo "skip: $OUT"
else
  echo "fetch: $URL"
  curl -fL --retry 3 -o "$OUT" "$URL"
  sha256sum "$OUT" >> "${DEST}/sources/sha256.txt"
fi
echo "ckpt_path=$OUT"
