# Ablation 10 — YOLOv11n on LaRS boxes from instance masks: training configuration and decisions

> **Role in the benchmark (2026-10-23).** YOLOv11s is the YOLO in the main comparison; this file documents the YOLOv11n runs (same configuration, `--model yolo11s.pt` for YOLOv11s, [TRAINING_FOLLOWUP.md](TRAINING_FOLLOWUP.md)). Training used Ultralytics 8.4.173, whose label assignment already enlarges boxes smaller than 16 px when choosing anchors (YOLO26's "STAL"), so these runs are not the original 2024 YOLOv11 recipe. See [fairness_fixes](../../fairness_fixes/README.md).

**Goal.** Train and evaluate the specialised detector (YOLOv11n) on LaRS itself, using boxes derived from the LaRS panoptic masks, so YOLO and EoMT can finally be compared on the same maritime data. Two checkpoints: standard full fine-tuning and a separate LoRA variant (scale / overfitting question).

Code: `water-debris-seg/scripts/{prepare_lars_yolo.py, train_yolo_lars.py, eval_yolo_lars.py}`, `water-debris-seg/water_debris_seg/yolo_lora.py`, job `water-debris-seg/scripts/snellius/ablation10_yolo_lars.sbatch`. Tests: `water-debris-seg/tests/test_yolo_lora.py`.

---

## 1. Boxes from masks

For each segment's instance mask M (pixels whose panoptic id equals the segment id) in `panoptic_masks/*.png`:

```
x_min = min{x : M(x,y)=1}    x_max = max{x : M(x,y)=1}
y_min = min{y : M(x,y)=1}    y_max = max{y : M(x,y)=1}
COCO box  = [x_min, y_min, w, h],  w = x_max − x_min + 1,  h = y_max − y_min + 1
YOLO line = "cls  cx/W  cy/H  w/W  h/H",  cx = x_min + w/2,  cy = y_min + h/2   (W×H = that image's size)
```

**Box-quality check** (`sources/lars_yolo_label_stats.json`, `sources/box_check_examples.png`):

| Check | Result |
|---|---|
| Traced box vs the box LaRS stores in `panoptic_annotations.json` | **0 mismatches** over all 2,803 images (max difference 0 px) |
| Written YOLO line decoded back to pixels | max error **0.001 px** (6-decimal rounding) |
| 5 rendered examples | Static Obstacle region, large boat, small buoy, float, "Other" — box tight on the annotated mask in every case |

Two properties to keep in mind: the box follows the *annotated* mask exactly (e.g. a buoy's reflection left out by the annotator is left out of the box), and an object split into several pieces by occlusion gets one box spanning all pieces (COCO convention).

## 2. Classes

| Decision | Choice | Why |
|---|---|---|
| Water, Sky | **Dropped** | Amorphous regions; a box over them carries no detection meaning |
| Static Obstacle | **Kept** as class 0 | Decision: keep obstacles. One box per Static Obstacle segment (as annotated), which usually spans the whole shoreline |
| Float | **Kept separate** (class 7) | Decision: do not merge with Other / objects, so debris remains visible as its own class |
| Object classes | 8 LaRS thing classes kept as annotated | Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other |

Class index: `0 Static Obstacle · 1 Boat/ship · 2 Row boats · 3 Paddle board · 4 Buoy · 5 Swimmer · 6 Animal · 7 Float · 8 Other`.

**Consequence of keeping Static Obstacle** (the EoMT failure concern): **662 of 1,082 val object boxes (61%) and 6,423 of 9,122 training object boxes (70%) have their centre inside the image's Static Obstacle box.** YOLO's NMS is per class, so obstacle boxes cannot suppress object boxes at inference, but the obstacle class still competes for the same anchors during training. To keep this visible, results are reported for all 9 classes **and** for the 8 object classes without Static Obstacle, plus per class.

**Crowd segments** (`iscrowd=1`: 199 train, 25 val, all boats/row boats) are left out of the YOLO labels (YOLO has no crowd flag) and kept as `iscrowd=1` in the COCO ground truth, so pycocotools ignores detections on them.

## 3. Splits and held-out protocol

LaRS frames come from videos; neighbouring frames are near-duplicates. The official LaRS train/val split shares **no** sequences (checked), and the held-out slice is built the same way.

| Split | Images | Sequences | Boxes | Role |
|---|---|---|---|---|
| `train_all` (official train) | 2,605 | 1,792 | 11,655 | stage B training |
| `train90` | 2,360 | 1,613 | 10,650 | stage A training |
| `held10` (10% of train *sequences*, seed 0) | 245 | 179 | 1,005 | stage A selection; monitored only in stage B |
| `val` (official val) | 198 | 77 | 1,268 | **final scoring only** |

LaRS test labels are not public (official evaluation server), so the official val split is the test set here, as for EoMT ablations 01–03.

**Two stages, per variant** (decision: use the held-out slice, but train the final model on the full train set including it):

1. **Stage A (selection).** Train on `train90` for a 100-epoch budget; validate on `held10` every epoch. The epoch with the best held-out fitness (Ultralytics' `0.1·mAP50 + 0.9·mAP50-95`) sets the epoch count. Held-out train/val loss curves are the overfitting check.
2. **Stage B (final).** Retrain from `yolo11n.pt` on `train_all` (held10 included) for the stage-A epoch count; keep **`last.pt`** (no checkpoint selection). `held10` is still validated each epoch for curves only (it is inside the training data now).
3. **Scoring.** Only the stage-B `last.pt` is scored on LaRS val. LaRS val is never used for training, early stopping or checkpoint choice.

This differs from the YOLO CANSURF run (ablation 08), whose best checkpoint was selected on the same split it was scored on (inflation ≈ 0.006 mAP); FloW followed the held-out pattern.

## 4. Model and hyper-parameters

| Setting | Value | Note |
|---|---|---|
| Model | YOLOv11n, COCO-pretrained `yolo11n.pt` | 448/499 tensors transferred; the 80-class head layers are re-initialised for 9 classes |
| Image size | **640** | Decision: keep 640 (minimum test size). See §6 |
| Epochs | stage A 100; stage B = stage-A best epoch | |
| Batch | 16 (nominal batch 64 via gradient accumulation ×4) | Ultralytics default `nbs=64` |
| Optimizer | AdamW, lr 0.000769, momentum 0.9, weight decay 5e-4 | Ultralytics `optimizer=auto` (chosen for 9 classes), unchanged for comparability |
| Schedule | 3 warm-up epochs, linear decay to lr×0.01, mosaic off for the last 10 epochs | Ultralytics defaults |
| Augmentation | Ultralytics defaults (mosaic, HSV, horizontal flip, scale 0.5, translate 0.1) | Same as ablation 08 |
| Precision | AMP (fp16) | |
| Seed | 0, `deterministic=True` | single seed |
| Tracking | W&B project `water-debris-seg`, group `ablation10_yolo_lars` | runs at `runs/<name>` |
| Loss | `7.5·box (CIoU) + 0.5·cls (BCE) + 1.5·DFL`; the total (`train/total_loss`, `val/total_loss`) was backfilled on W&B on the custom x-axis `epoch` | terms explained in [TRAINING_FOCAL.md §1b](TRAINING_FOCAL.md) |

## 5. Full fine-tuning vs LoRA (separate checkpoints)

| | Full (`lars_full_*`) | LoRA (`lars_lora_*`) |
|---|---|---|
| Trained | all 2.58M parameters | LoRA factors (206k) + detection head + BatchNorm = **651k of 2.81M (23%)** |
| Frozen | — | the 63 pretrained backbone/neck conv kernels |
| LoRA placement | — | every `Conv` block outside the `Detect` head; rank r = 8, alpha = 16 (scale 2) |

Why LoRA here: LaRS train is small (2,605 images, ~1,800 sequences) and dominated by boats; limiting trainable capacity is the standard way to test whether full fine-tuning overfits. The head is trained fully because its class layer is new.

**Implementation (Ultralytics has no LoRA):** `LoRAConv2d` keeps the pretrained kernel as a buffer, so Ultralytics' trainer cannot unfreeze it, and adds `(alpha/r)·(B·A)` reshaped to the kernel, with `B` zero-initialised (training starts exactly at the pretrained model). The kernel is computed in fp32 with autocast off and cast for the convolution, as AMP does for ordinary weights. Ultralytics' Conv+BN fusion assigns new weights onto the conv; `LoRAConv2d` intercepts that and folds the LoRA update in, so fused inference is exact. `eval_yolo_lars.py` also writes **`last_merged.pt`**, a plain Ultralytics checkpoint with the update merged into the kernels, which loads without this repository.

**Verified before training:** identical outputs to the pretrained model at initialisation; fused and merged models match the unfused one; checkpoint pickling round-trips; gradients reach the LoRA factors and never the frozen kernels; inside Ultralytics' trainer, all 126 LoRA tensors are in the optimizer. A 2-epoch GPU diagnostic showed both variants learning once AMP's loss scaler settled (loss 9.4 → ~5.8; weights and head move).

## 6. Image size and small objects

Decision: **640 px**, the minimum test size. LaRS mixes resolutions (val: 79 × 1920×1080, 71 × 1280×720, 15 × 2208×1242, smaller sizes down to 640×400; train mostly 1278×958), so the downscale to 640 depends on the source camera:

| Source width | Scale to 640 | A 16-px object becomes |
|---|---|---|
| 2208 | 0.29 | 4.6 px |
| 1920 | 0.33 | 5.3 px |
| 1278–1280 | 0.50 | 8 px |
| 640 | 1.00 | 16 px |

57% of val object boxes are smaller than 32×32 px (615 of 1,082) and 53 have a side under 4 px. Small-object recall is the expected weak point; AP_small and per-class recall are reported to quantify it.

## 7. Evaluation

`eval_yolo_lars.py` on LaRS val (198 images):

- **pycocotools** (headline; same scorer as ablations 04–09) on `sources/lars_val_gt_coco_boxes.json`: mAP, AP50, AP75, AP_small/medium/large, AR for **all 9 classes**, for the **8 object classes**, and **per class** (AP, AP50, AR100). Crowd boxes are ignored. Box area (w·h) defines small/medium/large, as in ablations 04–05.
- **Ultralytics val** on `eval_val.yaml` (its own mAP50-95, mAP50, precision, recall, per-class mAP); slightly different from pycocotools because crowd boxes are absent from YOLO labels and the AP interpolation differs.
- LoRA: the merged checkpoint is scored too and must give the same numbers.

## 8. Known limitations

- **Float is too rare to measure**: 22 training instances (all in `train90`, none in `held10`), 3 in val. Its per-class AP is reported but is not interpretable.
- **Static Obstacle boxes are huge** (often the full image width) and enclose most object centres; see §2.
- **Single seed**; no confidence intervals.
- **Stage-B curves on held10 are optimistic** (held10 is part of stage-B training); only stage A curves are used for selection and overfitting.

## 9. Reproduce

```bash
cd water-debris-seg
python scripts/prepare_lars_yolo.py                                   # labels, splits, GT, box check
VARIANT=full STAGE=A sbatch scripts/snellius/ablation10_yolo_lars.sbatch
VARIANT=lora STAGE=A sbatch scripts/snellius/ablation10_yolo_lars.sbatch
VARIANT=full STAGE=B sbatch --dependency=afterok:<A-full-job> scripts/snellius/ablation10_yolo_lars.sbatch   # trains, then scores last.pt
VARIANT=lora STAGE=B sbatch --dependency=afterok:<A-lora-job> scripts/snellius/ablation10_yolo_lars.sbatch
```

Runs of 2026-10-16: stage A full `28132643`, LoRA `28132645`; stage B full `28132656`, LoRA `28132657`.
