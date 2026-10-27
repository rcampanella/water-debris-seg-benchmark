#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-ab07-pflk
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_eomt_panoptic_flicker-%j.out

set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
VENV="${HOME}/envs/wdseg-eomt"
EOMT="${SCRATCH}/third_party/eomt"
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
OUT="${REPO}/eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
CKPT="${SCRATCH}/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/epoch=11-step=7812.ckpt"
mkdir -p "${SCRATCH}/logs" "$OUT/sources"

source "${VENV}/bin/activate"
source "${WDS}/scripts/snellius/wandb_env.sh"
export PYTHONPATH="${WDS}/scripts:${WDS}:${EOMT}:${PYTHONPATH:-}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ablation07_videomt_temporal}"
export WANDB_NAME="${WANDB_NAME:-eomt_panoptic_seq_flicker_${SLURM_JOB_ID}}"

python "${WDS}/scripts/lars_seq_eomt_flicker.py" \
  --task panoptic \
  --lightning-ckpt "${CKPT}" \
  --out "${OUT}/sources/eomt_panoptic_seq_flicker_${SLURM_JOB_ID}.json" \
  2>&1 | tee "${OUT}/sources/eomt_panoptic_seq_flicker_${SLURM_JOB_ID}.log"

python - <<PY
import json, os, wandb
p = "${OUT}/sources/eomt_panoptic_seq_flicker_${SLURM_JOB_ID}.json"
d = json.load(open(p))
wandb.init(project=os.environ.get("WANDB_PROJECT", "water-debris-seg"),
           group=os.environ.get("WANDB_RUN_GROUP"),
           name=os.environ.get("WANDB_NAME"),
           job_type="flicker_panoptic")
if d.get("mean_flicker_proxy") is not None:
    wandb.log({"flicker_panoptic/mean_proxy": d["mean_flicker_proxy"], "flicker_panoptic/n_clips": d["n_clips"]})
wandb.finish()
PY

echo "Ablation 07 EoMT panoptic seq flicker done."
