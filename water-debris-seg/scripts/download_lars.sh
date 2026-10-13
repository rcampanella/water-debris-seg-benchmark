#!/usr/bin/env bash
# LaRS v1.0.0 — https://lojzezust.github.io/lars-dataset/
set -euo pipefail

STILLS_ONLY=0
WITH_SEQUENCES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stills-only) STILLS_ONLY=1; shift ;;
    --with-sequences) WITH_SEQUENCES=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--stills-only] [--with-sequences]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

USER_NAME="${USER:-unknown}"
SCRATCH_ROOT="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER_NAME}/water-debris-seg}"
DEST="${WATER_DEBRIS_DATA_ROOT:-${SCRATCH_ROOT}/data}/lars"
mkdir -p "$DEST/sources"
cd "$DEST"

BASE="https://box.vicos.si/lars"
IMAGES="${BASE}/lars_v1.0.0_images.zip"
ANNS="${BASE}/lars_v1.0.0_annotations.zip"
SEQ="${BASE}/lars_v1.0.0_images_seq.zip"

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "skip (exists): $out"
    return
  fi
  echo "fetch: $url"
  curl -fL --retry 3 --continue-at - -o "$out" "$url"
  sha256sum "$out" >> sources/sha256.txt
}

download "$IMAGES" "lars_v1.0.0_images.zip"
download "$ANNS" "lars_v1.0.0_annotations.zip"

if [[ "$WITH_SEQUENCES" -eq 1 ]]; then
  download "$SEQ" "lars_v1.0.0_images_seq.zip"
fi

if [[ "$STILLS_ONLY" -eq 1 && "$WITH_SEQUENCES" -eq 0 ]]; then
  :
elif [[ "$WITH_SEQUENCES" -eq 0 ]]; then
  echo "Note: pass --with-sequences to fetch ~13.6 GB sequence zip (VidEoMT stage)."
fi

for z in lars_v1.0.0_images.zip lars_v1.0.0_annotations.zip; do
  [[ -f "$z" ]] || continue
  stamp="${z%.zip}.extracted"
  if [[ ! -f "$stamp" ]]; then
    unzip -q -o "$z"
    touch "$stamp"
  fi
done

if [[ -f lars_v1.0.0_images_seq.zip && ! -f lars_v1.0.0_images_seq.extracted ]]; then
  unzip -q -o lars_v1.0.0_images_seq.zip
  touch lars_v1.0.0_images_seq.extracted
fi

echo "LaRS tree under: $DEST"
ls -la "$DEST" | head -20
