#!/usr/bin/env bash
# CANSURF raw YOLO train/val — Zenodo v2 (images not in GitHub).
# DOI: https://doi.org/10.5281/zenodo.20100657
set -euo pipefail

USER_NAME="${USER:-unknown}"
SCRATCH_ROOT="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER_NAME}/water-debris-seg}"
DEST="${WATER_DEBRIS_DATA_ROOT:-${SCRATCH_ROOT}/data}/cansurf"
mkdir -p "$DEST/sources"
cd "$DEST"

ZIP="CANSURF-v2.zip"
URL="https://zenodo.org/records/20100759/files/${ZIP}?download=1"

if [[ ! -f "$ZIP" ]]; then
  echo "fetch: $URL"
  curl -fL --retry 3 --continue-at - -o "$ZIP" "$URL"
  sha256sum "$ZIP" >> sources/sha256.txt
fi

if [[ ! -f "${ZIP%.zip}.extracted" ]]; then
  unzip -q -o "$ZIP"
  touch "${ZIP%.zip}.extracted"
fi

# Drop augmented materialization if present (benchmark uses raw splits only).
if [[ -d augmentation_data ]]; then
  echo "removing augmentation_data/ (not used in benchmark)"
  rm -rf augmentation_data
fi

# Zenodo v2 unpacks train/ val/ data.yaml at archive root (no extra symlink).
if [[ ! -d train/images || ! -d val/images ]]; then
  echo "ERROR: expected train/images and val/images under $DEST"
  ls -la
  exit 1
fi

# Rewrite data.yaml paths for this host (upstream uses /content/...).
cat > data.yaml <<YAML
train: ${DEST}/train/images
val: ${DEST}/val/images
nc: 1
names: ['debris']
YAML

echo "CANSURF raw YOLO root: $DEST"
echo "train_images $(find train/images -type f | wc -l)"
echo "val_images $(find val/images -type f | wc -l)"
