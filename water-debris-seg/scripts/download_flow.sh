#!/usr/bin/env bash
# FloW-Img — ORCA application portal (no public direct URL). See ORCAUBOAT/FloW-Dataset README.
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
LOG="${REPO}/docs/water_debris_seg_decision_log.md"
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"
DEST="${SCRATCH}/data/flow"
MARKER="${DEST}/.flow_img_extracted"
mkdir -p "$DEST/sources"

if [[ -f "${MARKER}" ]]; then
  echo "FloW-Img already marked extracted: ${MARKER}"
  exit 0
fi

cat <<EOF
FloW-Img is not a single anonymous HTTP download. ORCA requires an application per sub-dataset.

GitHub (paper + pointer to portal):
  https://github.com/ORCAUBOAT/FloW-Dataset

Download portal (choose FloW-Img in sidebar, submit application):
  https://orca-tech.cn/en/datasets/FloW/Introduction

Support: datasets@orca-tech.com.cn

Manual steps:
  1. Apply on the portal for FloW-Img (vision boxes; skip FloW-RI unless needed).
  2. After approval, download and extract under: ${DEST}/FloW-Img/ (or match your layout)
  3. touch ${MARKER}

Benchmark target on scratch: ${DEST}/

Layout + sbatch order: ${REPO}/water-debris-seg/docs/FLOW_DATA.md
EOF

{
  echo "| $(date +%Y-%m-%d) | FloW-Img auto-download | Blocked — ORCA portal application |"
} >> "$LOG" 2>/dev/null || true
exit 2
