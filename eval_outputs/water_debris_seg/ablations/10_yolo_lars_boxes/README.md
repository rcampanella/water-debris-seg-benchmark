# Ablation 10 — YOLOv11n on LaRS boxes traced from instance masks

> **Role in the benchmark (2026-10-23).** **Main comparison table: YOLOv11s row** (LaRS box AP 0.176 / 0.195, ablation 10c). YOLOv11n (0.145 / 0.168), LoRA and focal-loss variants are references. YOLOv11s + SAM (LaRS PQ, mIoU) and the cross-domain FloW / CANSURF scores come from [fix 2](../../fairness_fixes/FIX2_YOLO_ON_LARS.md). Note: training used Ultralytics 8.4.173, whose label assignment already enlarges tiny boxes (YOLO26's "STAL"), so these are not the original 2024 YOLOv11 recipe. **LoRA is discarded for YOLO** (decision 2026-10-23): the LoRA runs here are references only; LoRA is compared for EoMT ([fix 4](../../fairness_fixes/FIX4_EOMT_LORA.md)).

Training configuration and every design decision: [TRAINING.md](TRAINING.md). This file holds the results.

**Setup in one line:** YOLOv11n (COCO-pretrained), 9 classes (Static Obstacle + 8 LaRS object classes, Float kept separate; water/sky dropped), 640 px. The epoch count was chosen on held-out train sequences (stage A), then the model was retrained on the full LaRS train set and its `last.pt` scored once on LaRS val (stage B). Two separate checkpoints: full fine-tuning and LoRA (r=8). No class reweighting (focal-loss variants: [§ 10b](#ablation-10b--focal-loss)).

## Results on LaRS val (198 images, pycocotools, crowd ignored)

| Checkpoint | mAP (9 cls) | AP50 | AP_small | AR100 | mAP (8 object cls) | AP50 | AP_small | AR100 | Ultralytics mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| Full fine-tune `last.pt` | **0.168** | **0.321** | **0.048** | **0.291** | **0.145** | **0.282** | **0.054** | **0.247** | 0.167 |
| LoRA `last.pt` | 0.153 | 0.281 | 0.029 | 0.284 | 0.127 | 0.233 | 0.033 | 0.235 | 0.162 |
| LoRA `last_merged.pt` (plain checkpoint) | 0.153 | 0.282 | 0.029 | 0.283 | 0.127 | 0.234 | 0.033 | 0.235 | 0.161 |

Sources: [`sources/eval_lars_full_stageB_28132656.json`](sources/eval_lars_full_stageB_28132656.json), [`sources/eval_lars_lora_stageB_28132657.json`](sources/eval_lars_lora_stageB_28132657.json) (each records the checkpoint and GT SHA-256). The merged LoRA checkpoint reproduces the LoRA checkpoint (`merged_matches: true`; the small differences come from fp16 storage).

### Per class (full fine-tune / LoRA)

| Class | Val boxes | AP | AP50 | AR100 |
|---|---|---|---|---|
| Static Obstacle | 186 | 0.355 / 0.359 | 0.636 / 0.665 | 0.642 / 0.677 |
| Boat/ship | 763 | 0.307 / 0.303 | 0.554 / 0.520 | 0.421 / 0.418 |
| Row boats | 32 | 0.233 / 0.214 | 0.406 / 0.361 | 0.394 / 0.384 |
| Paddle board | 30 | 0.339 / 0.261 | 0.549 / 0.392 | 0.527 / 0.480 |
| Buoy | 125 | 0.121 / 0.108 | 0.323 / 0.265 | 0.194 / 0.168 |
| Swimmer | 26 | 0.082 / 0.088 | 0.215 / 0.218 | 0.196 / 0.246 |
| Animal | 30 | 0.023 / 0.013 | 0.057 / 0.027 | 0.170 / 0.130 |
| Float | 3 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Other | 73 | 0.056 / 0.027 | 0.150 / 0.083 | 0.077 / 0.056 |

## Stage A: epoch selection and overfitting (train90 → held10)

| Variant | Best held10 epoch | held10 mAP50-95 at best / at 100 | Lowest held10 val loss (epoch) | Source |
|---|---|---|---|---|
| Full | 77 | 0.261 / 0.245 | epoch 96 | [`sources/epoch_selection_full.json`](sources/epoch_selection_full.json) |
| LoRA | 77 | 0.252 / 0.235 | epoch 67 | [`sources/epoch_selection_lora.json`](sources/epoch_selection_lora.json) |

held10 mAP50-95 at epochs 10/25/50/77/100: full 0.163/0.192/0.223/0.261/0.245, LoRA 0.160/0.201/0.228/0.252/0.235 (from `runs/lars_*_stageA/results.csv`). Neither variant shows clear overfitting: held-out loss stays flat to the end, and the mAP drop after epoch 77 falls in the last 10 epochs, where mosaic augmentation is switched off. Stage B therefore trained both variants for 77 epochs.

## Reading

- **Full fine-tuning beats LoRA** (objects mAP 0.145 vs 0.127; AP_small 0.054 vs 0.033). With 2.6k training images, the full 2.6M-parameter model does not overfit within 100 epochs, so limiting capacity with LoRA costs accuracy and gives nothing back here.
- **Small objects are the weak point**: AP_small 0.05 against overall object mAP 0.145. This matches the 640 px downscale analysis in [TRAINING.md §6](TRAINING.md).
- **Static Obstacle is the easiest class** (AP 0.36). Keeping it did not prevent object detection: per-class NMS means obstacle boxes do not suppress the objects inside them.
- **Gap from held-out to val**: held10 mAP50-95 ≈ 0.25 versus val ≈ 0.17 (Ultralytics metric, both). LaRS val comes from different sequences and contains more high-resolution (1920–2208 px) frames than train, which shrink objects further at 640 px.
- **Float is not measurable**: 3 val boxes, none detected; 22 training boxes, none in held10. See TRAINING.md §8.
- **Not comparable yet with EoMT on LaRS**: ablations 01–03 report PQ/mIoU from masks; no box mAP exists for EoMT on LaRS. YOLO gives boxes only, so PQ/mIoU cannot be computed for it.

## Ablation 10b — focal loss

Same protocol, focal classification loss (γ 1.5, α 0.25) instead of BCE; configuration in [TRAINING_FOCAL.md](TRAINING_FOCAL.md). Stage A chose 85 epochs (full) and 84 (LoRA): [`epoch_selection_full_focal.json`](sources/epoch_selection_full_focal.json), [`epoch_selection_lora_focal.json`](sources/epoch_selection_lora_focal.json).

| Checkpoint (LaRS val, pycocotools) | mAP (9) | AP50 (9) | mAP (8 obj) | AP50 (8 obj) | AP_small (obj) | AR100 (obj) | Source |
|---|---|---|---|---|---|---|---|
| Full, BCE (baseline) | **0.168** | **0.321** | **0.145** | **0.282** | **0.054** | **0.247** | [`eval_lars_full_stageB_28132656.json`](sources/eval_lars_full_stageB_28132656.json) |
| Full, focal | 0.158 | 0.278 | 0.129 | 0.229 | 0.038 | 0.230 | [`eval_lars_full_focal_stageB_28140699.json`](sources/eval_lars_full_focal_stageB_28140699.json) |
| LoRA, BCE (baseline) | 0.153 | 0.281 | 0.127 | 0.233 | 0.033 | 0.235 | [`eval_lars_lora_stageB_28132657.json`](sources/eval_lars_lora_stageB_28132657.json) |
| LoRA, focal | 0.154 | 0.265 | 0.124 | 0.209 | 0.030 | 0.230 | [`eval_lars_lora_focal_stageB_28140700.json`](sources/eval_lars_lora_focal_stageB_28140700.json) |

Per-class AP, BCE → focal (full / LoRA): Static Obstacle 0.355→0.393 / 0.359→0.400 · Boat/ship 0.307→0.310 / 0.303→0.300 · Row boats 0.233→0.242 / 0.214→0.171 · Paddle board 0.339→0.265 / 0.261→0.329 · Buoy 0.121→0.115 / 0.108→0.087 · Swimmer 0.082→0.046 / 0.088→0.064 · Animal 0.023→0.022 / 0.013→0.021 · Float 0 → 0 / 0 → 0 · Other 0.056→0.029 / 0.027→0.016.

**Reading.** Focal loss did not help. For full fine-tuning it lowered object mAP (0.145 → 0.129) and AP_small (0.054 → 0.038); for LoRA it was roughly neutral (0.127 → 0.124). The only consistent gain is on the large Static Obstacle class (+0.04 AP in both); the small and rare classes (Buoy, Swimmer, Other) went down, and Float stayed at 0 (3 val boxes). Single seed, no retuning of the cls gain for focal; the merged LoRA focal checkpoint reproduces its LoRA checkpoint (`merged_matches: true`).

Why focal loss did not help (loss and gradient by easy/hard positive/negative samples, detection images): [FOCAL_STUDY.md](FOCAL_STUDY.md).

W&B: [full focal A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_stageA_20261016_143757), [LoRA focal A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_focal_stageA_20261016_143757), [full focal B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_stageB_20261016_145813), [LoRA focal B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_focal_stageB_20261016_150405). Jobs 28140687 / 28140697 (A), 28140699 / 28140700 (B).

## Ablation 10c — follow-up runs

Same protocol; configuration, pre-set readings and sources in [TRAINING_FOLLOWUP.md](TRAINING_FOLLOWUP.md). LaRS val, 8 object classes (9 classes in brackets):

| Run | mAP obj (9 cls) | vs reference | Reading |
|---|---|---|---|
| YOLOv11s full, BCE | **0.176** (0.195) | full YOLOv11n 0.145 | YOLOv11n is capacity-limited |
| Full, focal α 0.75 | 0.143 (0.170) | focal α 0.25 0.129; BCE 0.145 | the α tilt explained most of focal's loss; still not better than BCE |
| LoRA r8, lr ×10 | 0.110 (0.139) | LoRA 0.127 | a ×10 global lr hurts (it also boosts head and BatchNorm) |
| LoRA r32 | 0.136 (0.161) | LoRA 0.127; full 0.145 | inconclusive: half the gap closed, below single-seed noise |

## Runs and artefacts

| | Full | LoRA |
|---|---|---|
| Stage A job / W&B | 28132643 · [lars_full_stageA_20261016_124920](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageA_20261016_124920) | 28132645 · [lars_lora_stageA_20261016_124920](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_stageA_20261016_124920) |
| Stage B job / W&B | 28132656 · [lars_full_stageB_20261016_130702](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageB_20261016_130702) | 28132657 · [lars_lora_stageB_20261016_131411](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_stageB_20261016_131411) |
| Scored checkpoint | `runs/lars_full_stageB/weights/last.pt` | `runs/lars_lora_stageB/weights/last.pt`, `last_merged.pt` |
| Logs | `sources/train_lars_full_stage{A,B}_*.log`, `sources/eval_lars_full_stageB_28132656.log` | `sources/train_lars_lora_stage{A,B}_*.log`, `sources/eval_lars_lora_stageB_28132657.log` |

W&B holds the training curves (held10 validation) only. The LaRS val scores above are in the eval JSONs, not on W&B. Checkpoints are on home (`/gpfs/home1`; `runs/<name>` links to `runs/runs/detect/water-debris-seg/<name>`), so they are safe from the scratch purge. "Val loss" in the stage-A table is box + cls + dfl loss on held10.
