#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -J wdseg-cansurf-paper
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/cansurf_paper_eval-%j.out

# CANSURF val with the CANSURF paper's detection protocol (RESULTS.md section "CANSURF paper protocol"), separate from
# our box AP. Scores the stored predictions of every main-table CANSURF cell, predicts with the two YOLO weights that
# have no stored predictions (CANSURF-trained stage B, COCO yolo11s zero-shot), and checks both against the paper's
# own script. ONLY=yolo skips the stored-prediction rows.
set -euo pipefail
REPO="${HOME}/projects/water-debris-seg-benchmark"
WDS="${REPO}/water-debris-seg"
S="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/sources"
J="${SLURM_JOB_ID}"
D_COCO="bottle,cup,frisbee,sports ball,kite,backpack,handbag,suitcase,umbrella"   # eval_eomt_fairness.py D_COCO
YOLO_CANSURF="${REPO}/eval_outputs/water_debris_seg/fairness_fixes/runs/cansurf_full_11s_stageB/weights/last.pt"
YOLO_COCO="${REPO}/data/water_debris/checkpoints/yolo/yolo11s.pt"
# shellcheck disable=SC1091
source "${HOME}/envs/wdseg-eomt/bin/activate"
export YOLO_CONFIG_DIR="${HOME}/.config/wdseg-ultralytics"
cd "${WDS}/scripts"

if [[ "${ONLY:-}" != yolo ]]; then
while read -r METHOD SRCNAME; do
  python cansurf_paper_metrics.py --method "${METHOD}" --preds "${S}/${SRCNAME}.preds.json" --scorer-json "${S}/${SRCNAME}.json"
done <<'EOF'
eomt_zeroshot_tightened fix1_coco133_cansurf_map-tightened_28511593
eomt_ab03 fix1_lars_panoptic_cansurf_28511595
eomt_lars_full fix1_lars_panoptic_cansurf_eomt_full_28547312
eomt_lars_lora fix1_lars_panoptic_cansurf_eomt_lora_28547314
eomt_lars_lora_e24 fix1_lars_panoptic_cansurf_eomt_lora_e24_28615360
eomt_lars_lora_e24_lr5 fix1_lars_panoptic_cansurf_eomt_lora_e24_lr5_28615362
eomt_cansurf_full fix1_box_instance_cansurf_eomt_full_28547307
eomt_cansurf_lora fix1_box_instance_cansurf_eomt_lora_28547309
yolo11s_lars fix2_yolo11s_lars_on_cansurf_28511597
EOF
fi

python cansurf_paper_metrics.py --method yolo11s_cansurf --weights "${YOLO_CANSURF}"
python cansurf_paper_metrics.py --method yolo11s_coco_zeroshot --weights "${YOLO_COCO}" --debris-classes "${D_COCO}"
python verify_cansurf_paper_metrics.py --weights "${YOLO_CANSURF}" --ours "${S}/cansurf_paper_yolo11s_cansurf_${J}.json"
python verify_cansurf_paper_metrics.py --weights "${YOLO_COCO}" --ours "${S}/cansurf_paper_yolo11s_coco_zeroshot_${J}.json" \
  --variant paper_class_id0
echo "cansurf_paper_eval done"
