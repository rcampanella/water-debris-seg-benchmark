# VidEoMT on Snellius (ablation 07)

Shared **EoMT venv** on scratch can host VidEoMT after extra deps (Detectron2 must be built with `--no-build-isolation` so it sees installed `torch`).

## Install

```bash
bash water-debris-seg/scripts/install_videomt_deps.sh
# or
sbatch water-debris-seg/scripts/snellius/setup_videomt_deps.sbatch
```

Smoke: `python water-debris-seg/scripts/videomt_env_smoke.py`

## LaRS sequences

- Stills: `data/lars/{train,val}/images`
- Clips: flat `images_seq/*.jpg` named `{scene}_seq_{id}_{frame:05d}.jpg`
- Inventory JSON: `eval_outputs/.../07_videomt_temporal/sources/lars_sequences_inventory.json` via `inventory_lars_sequences.py`

**LaRS YTVIS export (ablation 07):**

```bash
python water-debris-seg/scripts/export_lars_ytvis_json.py --split both
# JSON: $SCRATCH/data/lars/ytvis/lars_ytvis_{train,val}.json
sbatch water-debris-seg/scripts/snellius/ablation07_lars_ytvis_export.sbatch
sbatch water-debris-seg/scripts/snellius/ablation07_lars_ytvis_load_smoke.sbatch  # after export
```

Registration: `water_debris_seg/videomt/lars_ytvis.py` → Detectron2 names `lars_ytvis_train`, `lars_ytvis_val` (thing panoptic labels on keyframes only).

**Train (ablation 07):**

```bash
sbatch water-debris-seg/scripts/snellius/ablation07_videomt_train.sbatch
sbatch water-debris-seg/scripts/snellius/ablation07_videomt_train_16k.sbatch   # longer schedule
```

**Flicker (val `images_seq`, from saved `results.json`):**

```bash
sbatch water-debris-seg/scripts/snellius/ablation07_videomt_seq_flicker.sbatch [results.json] [label]
```

Config: `configs/ablation07_videomt_lars_vitb.yaml` (ViT-B online, 8 classes, 4k iters). Set **`MODEL.SEM_SEG_HEAD.NUM_CLASSES`** to the same value as **`MODEL.BACKBONE.NUM_CLASSES`** — `YTVISDatasetMapper` uses the sem-seg head count for padding instances; leaving the YTVIS default (54) poisons mask/class loss with label 54. LaRS keyframes are sparse: `run_videomt_train_lars.py` applies `water_debris_seg/videomt/prepare_targets_patch.py` so video labels come from any labeled frame, not only the last sampled frame (avoids dummy class 8). It also applies `water_debris_seg/videomt/keyframe_patch.py`: (1) training windows always contain a labeled keyframe — upstream `randrange(L-5)` never reaches index 9 of the 10-frame LaRS clips, so ~92% of iterations had no GT and zero mask loss; (2) the loss is computed only on labeled frames, because `id == -1` means "not annotated" on LaRS, not "absent". (3) `apply_keyframe_eval_patch` scores VIS AP **per labeled keyframe**: export tracks are single-frame (LaRS has no cross-keyframe IDs), and upstream video IoU counts predictions on GT-less frames as union, so upstream AP is 0 even for a perfect tracker. Offline: `scripts/rescore_videomt_ytvis_keyframes.py --results <run>/inference/results.json` (`--oracle exact` must give AP 1.0). Eval-only: `sbatch scripts/snellius/ablation07_videomt_eval.sbatch <ckpt> [MAX_NUM=50]`. Run `scripts/videomt_one_batch_loss_smoke.py` (sbatch `ablation07_videomt_loss_smoke.sbatch`) before any retrain: it must print PASS. Entry: `scripts/run_videomt_train_lars.py` (registers LaRS + W&B project `water-debris-seg`).

## Eval-only smoke (official weights)

After deps + `DETECTRON2_DATASETS` points at a supported set, see [VidEoMT README](https://github.com/tue-mps/videomt) `train_net_video.py --eval-only`.

LaRS train/eval remains custom adapter work (ablation 07).
