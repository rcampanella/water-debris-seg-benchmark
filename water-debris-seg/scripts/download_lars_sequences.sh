#!/usr/bin/env bash
# LaRS sequence zip (~13.6 GB) — VidEoMT / ablation 07.
set -euo pipefail
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"
DEST="${WATER_DEBRIS_DATA_ROOT:-${SCRATCH}/data}/lars"
mkdir -p "$DEST/sources"
cd "$DEST"
URL="https://box.vicos.si/lars/lars_v1.0.0_images_seq.zip"
ZIP="lars_v1.0.0_images_seq.zip"
if [[ ! -f "$ZIP" ]]; then
  curl -fL --retry 3 --continue-at - -o "$ZIP" "$URL"
  sha256sum "$ZIP" >> sources/sha256.txt
fi
if [[ ! -f lars_v1.0.0_images_seq.extracted ]]; then
  unzip -q -o "$ZIP"
  touch lars_v1.0.0_images_seq.extracted
fi
echo "LaRS sequences under $DEST"
