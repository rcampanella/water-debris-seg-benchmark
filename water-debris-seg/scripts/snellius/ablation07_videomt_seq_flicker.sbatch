#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH -J wdseg-ab07-vmt-flk
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/ablation07_videomt_seq_flicker-%j.out

# VidEoMT temporal flicker on val images_seq (from saved VIS results.json).
# Usage: sbatch ablation07_videomt_seq_flicker.sbatch [results.json] [ckpt_label]
set -euo pipefail
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
WDS="${HOME}/projects/water-debris-seg-benchmark/water-debris-seg"
OUT="${WDS}/../eval_outputs/water_debris_seg/ablations/07_videomt_temporal"
REF="${OUT}/sources/eomt_semantic_seq_flicker_27711595.json"
RESULTS="${1:-${SCRATCH}/artifacts/videomt/eval_27775358/inference/results.json}"
LABEL="${2:-27775358}"
mkdir -p "${SCRATCH}/logs" "${OUT}/sources"

source "${SCRATCH}/envs/eomt/bin/activate"
export PYTHONPATH="${WDS}/scripts:${WDS}:${PYTHONPATH:-}"

python "${WDS}/scripts/lars_seq_videomt_flicker.py" \
  --results "${RESULTS}" \
  --reference-clips "${REF}" \
  --out "${OUT}/sources/videomt_seq_flicker_${SLURM_JOB_ID}.json" \
  2>&1 | tee "${OUT}/sources/videomt_seq_flicker_${SLURM_JOB_ID}.log"

python - <<PY
import json, os, wandb
p = "${OUT}/sources/videomt_seq_flicker_${SLURM_JOB_ID}.json"
d = json.load(open(p))
wandb.init(project=os.environ.get("WANDB_PROJECT", "water-debris-seg"),
           group=os.environ.get("WANDB_RUN_GROUP", "ablation07_videomt_temporal"),
           name=os.environ.get("WANDB_NAME", "videomt_seq_flicker_${SLURM_JOB_ID}"),
           job_type="flicker_videomt")
if d.get("mean_flicker_proxy") is not None:
    wandb.log({"flicker/mean_proxy": d["mean_flicker_proxy"], "flicker/n_clips": d["n_clips"],
               "flicker/source_results": "${LABEL}"})
wandb.finish()
PY

echo "VidEoMT seq flicker done (results=${LABEL})."
