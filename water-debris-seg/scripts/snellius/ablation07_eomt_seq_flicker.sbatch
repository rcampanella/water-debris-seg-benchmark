#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab07-flk
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_eomt_seq_flicker-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${SCRATCH}/envs/eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/mk2w7dr8/checkpoints/epoch=30-step=2356.ckpt"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation07_videomt_temporal}"
export WANDB_NAME="${WANDB_NAME:-eomt_seq_flicker_${SLURM_JOB_ID}}"

python "${WDS}/scripts/summarize_lars_attributes.py" \
  --out "${OUT}/sources/lars_val_attributes.json"

python "${WDS}/scripts/lars_seq_eomt_flicker.py" \
  --lightning-ckpt "${CKPT}" \
  --out "${OUT}/sources/eomt_semantic_seq_flicker_${SLURM_JOB_ID}.json" \
  2>&1 | tee "${OUT}/sources/eomt_seq_flicker_${SLURM_JOB_ID}.log"

python - <<PY
import json, os, wandb
p = "${OUT}/sources/eomt_semantic_seq_flicker_${SLURM_JOB_ID}.json"
d = json.load(open(p))
wandb.init(project=os.environ.get("WANDB_PROJECT", "water-debris-seg"),
           group=os.environ.get("WANDB_RUN_GROUP"),
           name=os.environ.get("WANDB_NAME"),
           job_type="flicker_baseline")
if d.get("mean_flicker_proxy") is not None:
    wandb.log({"flicker/mean_proxy": d["mean_flicker_proxy"], "flicker/n_clips": d["n_clips"]})
wandb.finish()
PY

echo "Ablation 07 EoMT seq flicker baseline done."
