#!/usr/bin/env bash
set -euo pipefail
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"
TP="${SCRATCH}/third_party"
mkdir -p "$TP"

clone_one() {
  local url="$1" dir="$2"
  if [[ -d "$TP/$dir/.git" ]]; then
    echo "skip: $dir"
    return
  fi
  git clone --depth 1 "$url" "$TP/$dir"
}

clone_one https://github.com/tue-mps/eomt.git eomt
clone_one https://github.com/tue-mps/videomt.git videomt
clone_one https://github.com/ultralytics/ultralytics.git ultralytics

echo "third_party under $TP"
