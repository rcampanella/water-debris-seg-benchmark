# Ablation 10b — YOLOv11n on LaRS boxes with focal classification loss (full and LoRA)

**Question.** Does a focal classification loss change what YOLO learns on LaRS, especially for the rare classes (Float: 22 training boxes out of 11,655)? Compared against the ablation 10 baseline, which uses Ultralytics' default BCE ([README.md](README.md), [TRAINING.md](TRAINING.md)).

**Everything is identical to the baseline except the classification loss.** Same data, labels, splits, two-stage protocol, model, image size, optimizer, schedule, augmentation and seed. Two new checkpoints: full fine-tuning + focal, and LoRA + focal.

---

## 1. The change: focal loss on the class scores

Ultralytics' detection loss sums a per-anchor, per-class BCE on the class logits, divided by the sum of the target scores. The focal variant replaces only that element-wise BCE:

```
p    = sigmoid(logit)                 t = target score (YOLO's soft, IoU-aligned target in [0,1])
p_t  = t·p + (1 − t)·(1 − p)
α_t  = t·α + (1 − t)·(1 − α)
FL   = α_t · (1 − p_t)^γ · BCE(logit, t)          γ = 1.5, α = 0.25
cls_loss = Σ FL / Σ t                              (same normalisation as the baseline), × cls gain 0.5
```

| Choice | Value | Why |
|---|---|---|
| γ (focusing) | **1.5** | Ultralytics' `FocalLoss` default and YOLOv5's `fl_gamma` convention |
| α (balance) | **0.25** | Same defaults (RetinaNet pairing with γ) |
| Soft targets | used as is in `p_t`, `α_t` | As YOLOv5's focal loss did; no varifocal reformulation |
| Box / DFL losses | unchanged | Only classification is affected |
| cls gain, lr, epochs | unchanged (0.5, auto AdamW, stage-A selection) | Isolates the loss change; no retuning |
| `cls_pw` class weights | off (0.0), as in the baseline | One change at a time |

**What focal loss does and does not target.** It down-weights predictions that are already easy, which are overwhelmingly background anchors (an easy negative at logit −6 keeps < 0.1% of its BCE). It does **not** up-weight a class by how rare it is. Any gain on Float comes indirectly, from training no longer being dominated by easy background. Inverse-frequency weighting (`cls_pw`) would address class frequency directly; it is not part of this run.

**Expected side effect.** With α = 0.25 and the modulating factor, the classification loss is numerically much smaller than BCE at the same gain, so `cls_loss` values on W&B are **not comparable** with the baseline runs. Compare mAP and recall, not loss values.

## 1b. The total loss and its terms (what to watch on W&B)

Ultralytics logs three terms, each **already multiplied by its gain**; the total is their sum and is what the optimizer minimises (per image, averaged over the epoch's batches for `train/…`, over the held10 batches for `val/…`):

```
total_loss = box_loss + cls_loss + dfl_loss
           = 7.5 · L_CIoU  +  0.5 · L_cls  +  1.5 · L_DFL
```

All three are normalised by the same quantity, the sum of the soft target scores `Σt` over the batch. Which anchors are positive, and their targets `t`, come from Ultralytics' **TaskAlignedAssigner** (top-10 anchors per object, ranked by `score^0.5 · IoU^6`; `t` is that alignment metric rescaled so the best anchor of each object gets its best IoU).

| Logged term | What it measures | Formula (positives = foreground anchors) | Gain | Changed by focal? |
|---|---|---|---|---|
| `box_loss` | Box overlap and shape | `Σ_pos w · (1 − CIoU(pred, gt)) / Σt`, `w` = the anchor's target score. CIoU = IoU − (centre distance² / enclosing-box diagonal²) − aspect-ratio penalty | 7.5 | No |
| `cls_loss` | Class scores, for **every** anchor × 9 classes (background = all targets 0) | Baseline: `Σ BCE(logit, t) / Σt`. **Focal: `Σ α_t (1 − p_t)^γ BCE(logit, t) / Σt`** (§1) | 0.5 | **Yes** |
| `dfl_loss` | Box-edge precision (Distribution Focal Loss) | Each of the 4 box edges is predicted as a distribution over 16 distance bins; DFL is the cross-entropy against the two bins around the true distance (linear interpolation weights), weighted by `w`, `/ Σt` | 1.5 | No |
| `total_loss` | Sum of the three | — | — | Yes, through `cls_loss` |

**What to watch, in order:**

1. `metrics/mAP50-95(B)` on held10: the stage-A epoch selection uses this (fitness = 0.1·mAP50 + 0.9·mAP50-95).
2. `val/total_loss` against `train/total_loss`: a rising validation total while training falls is the overfitting signal. **Within a run only.** The focal total is lower by construction (its `cls_loss` is numerically smaller), so focal and BCE totals cannot be compared.
3. `val/box_loss` and `val/dfl_loss`: unchanged by focal loss, so these **can** be compared across focal and BCE runs.

**How the total is logged.** `train_yolo_lars.py` adds `train/total_loss`, `val/total_loss` and `epoch` to the logged metrics every epoch (`add_total_loss`). It was added during this ablation: the 8 runs already started (4 baseline, 4 focal) were **backfilled** from their `results.csv` with `water-debris-seg/scripts/wandb_backfill_total_loss.py`. On those runs the total sits on W&B's custom x-axis `epoch` (equal to the original step), because W&B cannot write into past steps; set a panel's x-axis to `epoch` to compare all runs. `results.csv` itself has only the three terms.

## 2. Implementation

| File | Role |
|---|---|
| `water-debris-seg/water_debris_seg/yolo_focal.py` | `FocalBCE` (element-wise focal term); `FocalDetectionModel` (a `DetectionModel` whose criterion uses it); `use_focal()`; `to_plain_detection_model()` |
| `water-debris-seg/scripts/train_yolo_lars.py` | `--loss focal [--focal-gamma 1.5 --focal-alpha 0.25]`; one trainer subclass applies focal and/or LoRA right after the pretrained weights load |
| `water-debris-seg/scripts/eval_yolo_lars.py` | Loads focal checkpoints; the merged LoRA checkpoint is converted back to a plain `DetectionModel` |
| `water-debris-seg/scripts/snellius/ablation10_yolo_lars.sbatch` | `LOSS=focal` → run names `lars_<variant>_focal_stage<A/B>` |
| `water-debris-seg/scripts/wandb_backfill_total_loss.py` | Writes `train/total_loss`, `val/total_loss` (x-axis `epoch`) to runs that predate `add_total_loss` |
| `water-debris-seg/tests/test_yolo_focal.py` | Formula check; the criterion really uses focal and the class branch gets gradient; focal + LoRA together; checkpoint pickling and plain conversion |

The model's class is switched before Ultralytics creates the EMA copy, so training loss, held-out validation loss and saved checkpoints all use the focal criterion. Ultralytics has no focal option for detection: its `FocalLoss` class exists but is not used by `v8DetectionLoss`.

## 3. Configuration (both runs)

| Setting | Full + focal | LoRA + focal |
|---|---|---|
| Model | YOLOv11n from `yolo11n.pt`, 9 classes | same |
| Trainable | all 2.58M | LoRA r=8, α_LoRA=16 on 63 backbone/neck convs + head + BatchNorm (651k of 2.81M) |
| Loss | box (CIoU) 7.5 + DFL 1.5 + **focal cls** 0.5 | same |
| Data | `train_all` (stage B) / `train90`→`held10` (stage A), 9 classes, crowd excluded | same |
| Image size / batch | 640 / 16 (nbs 64) | same |
| Optimizer | AdamW lr 0.000769, momentum 0.9, wd 5e-4 (`optimizer=auto`) | same |
| Schedule | stage A 100 epochs; stage B = best held10 epoch of **this variant's own** stage A; 3 warm-up epochs; mosaic off for last 10 | same |
| Precision / seed | AMP, seed 0, deterministic | same |
| Scored | stage-B `last.pt` on LaRS val (pycocotools: 9 classes, 8 object classes, per class) | + `last_merged.pt` |

## 4. Runs and W&B

All runs log to W&B project **`water-debris-seg`**, group **[`ablation10_yolo_lars`](https://wandb.ai/riccardo-campanella/water-debris-seg/groups/ablation10_yolo_lars)**, next to the baseline runs. Ultralytics names each W&B run `<name>_<YYYYMMDD_HHMMSS>`; Slurm jobs submitted 2026-10-16 (a first submission, jobs 28140501/28140510/28140512/28140516, failed in 8 s before training because the job variables were not passed; resubmitted with `--export`) (stage B chained `afterok` on its stage A). Results: [README.md](README.md#ablation-10b--focal-loss).

| Run name | Stage | Slurm job | W&B run |
|---|---|---|---|
| `lars_full_focal_stageA` | A: train90 → held10, 100 epochs | 28140687 | [lars_full_focal_stageA_20261016_143757](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_stageA_20261016_143757) |
| `lars_lora_focal_stageA` | A | 28140697 | [lars_lora_focal_stageA_20261016_143757](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_focal_stageA_20261016_143757) |
| `lars_full_focal_stageB` | B: train_all, own best epoch, then LaRS val scoring | 28140699 | [lars_full_focal_stageB_20261016_145813](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_stageB_20261016_145813) |
| `lars_lora_focal_stageB` | B | 28140700 | [lars_lora_focal_stageB_20261016_150405](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_focal_stageB_20261016_150405) |

Baseline (BCE) runs for comparison: [lars_full_stageA](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageA_20261016_124920), [lars_lora_stageA](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_stageA_20261016_124920), [lars_full_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageB_20261016_130702), [lars_lora_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_stageB_20261016_131411).

W&B holds training curves and held10 validation. LaRS val scores go to `sources/eval_lars_<variant>_focal_stageB_<job>.json` (as for the baseline), not to W&B.

**Outputs:** `runs/lars_<variant>_focal_stage{A,B}/`, `sources/epoch_selection_<variant>_focal.json`, `sources/train_*_focal_*.log`, `sources/eval_*_focal_*.{json,log}`.

**Expected time:** about 18 min (stage A) + 15 min (stage B) for full, 25 + 21 min for LoRA (baseline timings); both variants in parallel, so about 46 min plus queue.

## 5. How the result will be read

- Primary comparison: focal vs BCE, same variant, on LaRS val: mAP for 9 classes and for the 8 object classes, AP_small, AR100, and per-class AP/AR (especially the small and rare classes: Buoy, Swimmer, Animal, Other).
- **Float remains unmeasurable on LaRS** (3 val boxes, none in held10). A Float effect would need a cross-dataset check (e.g. FloW test); not part of this run.
- **Single seed.** The baseline full-vs-LoRA gap is 0.018 object mAP; a focal effect smaller than that cannot be separated from seed noise without more seeds.

## 6. Reproduce

```bash
cd water-debris-seg
sbatch --export=ALL,LOSS=focal,VARIANT=full,STAGE=A scripts/snellius/ablation10_yolo_lars.sbatch
sbatch --export=ALL,LOSS=focal,VARIANT=lora,STAGE=A scripts/snellius/ablation10_yolo_lars.sbatch
sbatch --export=ALL,LOSS=focal,VARIANT=full,STAGE=B --dependency=afterok:<A-full-focal-job> scripts/snellius/ablation10_yolo_lars.sbatch
sbatch --export=ALL,LOSS=focal,VARIANT=lora,STAGE=B --dependency=afterok:<A-lora-focal-job> scripts/snellius/ablation10_yolo_lars.sbatch
```
