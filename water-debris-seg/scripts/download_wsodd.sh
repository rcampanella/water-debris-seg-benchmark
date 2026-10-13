#!/usr/bin/env bash
# WSODD — OUT OF SCOPE for benchmark v1 (2026-10-07). See ablation 06 README + decision log.
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
SCRATCH="${WATER_DEBRIS_SCRATCH_ROOT:-/scratch-shared/${USER}/water-debris-seg}"

cat <<EOF
WSODD is excluded from water-debris-seg benchmark v1.

Rationale: dataset is on Baidu Pan only (https://github.com/sunjiaen/WSODD#dataset_wsodd).
Ablation 06 cancelled. Use LaRS + FloW + CANSURF for the published benchmark.

Decision: ${REPO}/docs/water_debris_seg_decision_log.md (2026-10-07)

If you obtain WSODD manually for a side experiment, target: ${SCRATCH}/data/wsodd/
EOF
exit 3
