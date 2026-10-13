#!/usr/bin/env bash
# Source before train/eval jobs. Set WANDB_API_KEY in ~/.config/water-debris-seg/wandb.env
set -euo pipefail
WANDB_ENV="${HOME}/.config/water-debris-seg/wandb.env"
if [[ -f "$WANDB_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$WANDB_ENV"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
else
  export WANDB_MODE="${WANDB_MODE:-online}"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-water-debris-seg}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_DIR="${WANDB_DIR:-/scratch-shared/${USER}/water-debris-seg/artifacts/wandb}"
mkdir -p "$WANDB_DIR"
