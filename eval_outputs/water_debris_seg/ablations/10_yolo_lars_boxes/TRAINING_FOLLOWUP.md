# Ablation 10c — follow-up runs: why LoRA trails full fine-tuning, model size, and focal α

**Questions.**
1. LoRA trails full fine-tuning (objects mAP 0.127 vs 0.145). Is LoRA **under-trained** (its learning rate was copied from full fine-tuning) or **capacity-limited** (rank 8)?
2. Is full fine-tuning limited by the **model size** (YOLOv11n is the smallest YOLOv11)?
3. Did focal loss lose because **α = 0.25 weights objects 3× less than background** ([FOCAL_STUDY.md § 6](FOCAL_STUDY.md#6-interpretation))?

**Everything not listed below is identical to the baseline** ([TRAINING.md](TRAINING.md)): data, 9 classes, splits, two-stage protocol (stage A on train90 → held10 picks the epoch count; stage B retrains on train_all and keeps `last.pt`; LaRS val scored once), 640 px, batch 16, schedule, augmentation, AMP, seed 0. Each run picks its own epoch count in its own stage A.

## The four runs

| Run tag | Starts from | Changed vs its reference | Reference run | Tests |
|---|---|---|---|---|
| `lora_lr10` | LoRA, BCE | learning rate **0.00769** (×10) | LoRA BCE | Q1: optimization |
| `lora_r32` | LoRA, BCE | rank **r = 32**, α_LoRA = 64 (same scale α/r = 2) | LoRA BCE | Q1: capacity |
| `full_11s` | Full, BCE | model **YOLOv11s** (`yolo11s.pt`, 9.46M params vs 2.62M) | Full BCE | Q2 |
| `full_focal_a075` | Full, focal γ 1.5 | focal **α = 0.75** (objects weighted 3× *more* than background) | Full focal α 0.25, and Full BCE | Q3 |

### Details

| Setting | Value |
|---|---|
| Learning rate, `lora_lr10` | Explicit `optimizer=AdamW, lr0=0.00769, momentum=0.9, warmup_bias_lr=0.0`. This is exactly what Ultralytics' `optimizer=auto` chose for the baseline (AdamW, lr 0.002·5/(4+9) = 0.000769, momentum 0.9, no bias warm-up), with only the lr ×10. Weight decay, warm-up epochs and the decay to lr×0.01 unchanged |
| Learning rate, other runs | `optimizer=auto` as in the baseline (AdamW 0.000769) |
| LoRA rank, `lora_r32` | r = 32, α_LoRA = 64: the update scale α/r = 2 stays as in the baseline, so only the rank changes. LoRA factors 4× larger: about 825k instead of 206k (logged by the job) |
| YOLOv11s weights | Ultralytics release v8.4.0 `yolo11s.pt`, SHA-256 `85a76fe8…d502d5`, stored in `runs/yolo11s.pt` |
| Focal, `full_focal_a075` | γ 1.5 as before; α_t = 0.75 for object samples, 0.25 for background. Class-loss weight 0.5 unchanged |

## What will count as an answer (decided before the runs)

Reference values, LaRS val, pycocotools, 8 object classes: full BCE **0.145**, LoRA BCE **0.127** (gap 0.018), full focal α 0.25 **0.129**.

| Result | Reading |
|---|---|
| `lora_lr10` ≥ 0.140 and `lora_r32` < 0.135 | LoRA was **under-trained**: optimization, not capacity |
| `lora_r32` ≥ 0.140 and `lora_lr10` < 0.135 | LoRA was **capacity-limited** (rank 8 too small) — your "smaller manifold" explanation |
| Both ≥ 0.140 | Both matter |
| Neither ≥ 0.135 | Neither explains it; the frozen backbone kernels themselves are the limit |
| `full_11s` > 0.155 | YOLOv11n is **capacity-limited** |
| `full_focal_a075` ≥ 0.140 | The **α tilt** explained most of focal's loss; < 0.135: focal does not help YOLOv11 here regardless of α |

Single seed per run: differences under about 0.01 are treated as inconclusive. LoRA lr ×10 can also train unstably; if stage A shows divergence (rising losses), that is reported as such.

## Runs, W&B and outputs

All runs log to W&B project `water-debris-seg`, group [`ablation10_yolo_lars`](https://wandb.ai/riccardo-campanella/water-debris-seg/groups/ablation10_yolo_lars), with `train/total_loss` and `val/total_loss` logged natively (x-axis `epoch`). Exact W&B run links and Slurm job IDs are added below when the jobs start.

| Run name | Slurm job (A / B) | W&B run |
|---|---|---|
| `lars_lora_lr10_stageA` / `_stageB` | 28378803 / 28378809 | A: [lars_lora_lr10_stageA_20261020_161452](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_lr10_stageA_20261020_161452); B: [lars_lora_lr10_stageB_20261020_165808](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_lr10_stageB_20261020_165808) |
| `lars_lora_r32_stageA` / `_stageB` | ~~28378804 / 28378810~~ → **28409802 / 28409804** | A: [lars_lora_r32_stageA_20261021_104824](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_r32_stageA_20261021_104824) (failed first attempt: lars_lora_r32_stageA_20261020_161452, no training data); B: [lars_lora_r32_stageB_20261021_115911](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_r32_stageB_20261021_115911) |
| `lars_full_11s_stageA` / `_stageB` | 28378805 / 28378811 | A: [lars_full_11s_stageA_20261020_161452](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_11s_stageA_20261020_161452); B: [lars_full_11s_stageB_20261020_170700](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_11s_stageB_20261020_170700) |
| `lars_full_focal_a075_stageA` / `_stageB` | 28378808 / 28378812 | A: [lars_full_focal_a075_stageA_20261020_161452](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_a075_stageA_20261020_161452); B: [lars_full_focal_a075_stageB_20261020_165813](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_a075_stageB_20261020_165813) |

Settings confirmed in the stage-A logs: `lora_lr10` args `optimizer=AdamW, lr0=0.00769, momentum=0.9, warmup_bias_lr=0.0`; `lora_r32` `[LoRA] r=32 alpha=64.0`, 825,472 LoRA / 1,270,459 trainable parameters; `full_11s` `model=yolo11s.pt`; `full_focal_a075` `[focal] gamma=1.5 alpha=0.75`.

Outputs per run: `runs/lars_<tag>_stage{A,B}/` (with `run_spec.json` recording model, lr, tag, LoRA rank), `sources/epoch_selection_<tag>.json`, `sources/train_lars_<tag>_*.log`, `sources/eval_lars_<tag>_stageB_<job>.{json,log}`.

**Expected time** (stage A + B, from the baseline timings): LoRA runs ~46 min each, focal α 0.75 ~35 min, YOLOv11s ~1–1.5 h (3.6× the parameters; not timed before). All four in parallel: ~1.5 h plus queue.

**Rank-32 rerun.** The first `lora_r32` stage A (28378804) crashed after ~10 min, before training: the four jobs started together and all rebuilt the shared `train90/labels.cache`; one deleted the file while another was replacing it (`FileNotFoundError` in Ultralytics' `save_dataset_cache_file`, Slurm log `ablation10_yolo_lars-28378804.out`). Its stage B (28378810) was cancelled by the dependency. Resubmitted alone on 2026-10-21 as 28409802 / 28409804 (cache already built). The failed W&B run [lars_lora_r32_stageA_20261020_161452](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_r32_stageA_20261020_161452) holds no training data.

## Results (LaRS val, pycocotools, stage-B `last.pt`)

| Run | Epochs (stage A best) | mAP 9 cls | **mAP 8 obj** | AP50 obj | AP_small obj | AR100 obj | Reference (obj mAP) | Pre-set reading | Source |
|---|---|---|---|---|---|---|---|---|---|
| `full_11s` | 90 | **0.195** | **0.176** | **0.308** | **0.062** | **0.251** | full BCE 0.145 | > 0.155 → **YOLOv11n is capacity-limited** | [`eval_lars_full_11s_stageB_28378811.json`](sources/eval_lars_full_11s_stageB_28378811.json) |
| `full_focal_a075` | 83 | 0.170 | 0.143 | 0.251 | 0.046 | 0.237 | focal α 0.25: 0.129; BCE 0.145 | ≥ 0.140 → **the α tilt explained most of focal's loss**; still not better than BCE (−0.002, inconclusive) | [`eval_lars_full_focal_a075_stageB_28378812.json`](sources/eval_lars_full_focal_a075_stageB_28378812.json) |
| `lora_lr10` | 99 | 0.139 | 0.110 | 0.216 | 0.038 | 0.207 | LoRA 0.127 | < 0.135 → **not under-training at this lr**; ×10 made it worse | [`eval_lars_lora_lr10_stageB_28378809.json`](sources/eval_lars_lora_lr10_stageB_28378809.json) |
| `lora_r32` | 58 | 0.161 | 0.136 | 0.262 | 0.043 | 0.240 | LoRA 0.127 | between the thresholds (≥ 0.135, < 0.140) and +0.009 < 0.01 noise: **inconclusive**. Rank closes about half the gap to full fine-tuning; not decisive at one seed | [`eval_lars_lora_r32_stageB_28409804.json`](sources/eval_lars_lora_r32_stageB_28409804.json) |

Stage-A selections: `sources/epoch_selection_{full_11s,full_focal_a075,lora_lr10,lora_r32}.json` (held10 mAP50-95 at best: 0.294, 0.258, 0.229, 0.242; baselines full 0.261, LoRA 0.252). LoRA merged checkpoints reproduce their LoRA checkpoints (`merged_matches: true`).

**Caveat on `lora_lr10`.** The ×10 learning rate applies to *every* trainable parameter, including the fully trained detection head and BatchNorm, not only the LoRA factors. Early training was clearly disturbed (held10 val total loss 6.5 vs 3.9 at epoch 4; mAP 0.05 vs 0.12) and it never caught up. So the run shows that a ×10 global lr hurts; it does not rule out that a higher lr *on the LoRA factors only* would help. That cleaner test needs separate optimizer groups.

**Answer to Q1 (LoRA vs full).** A ×10 global learning rate made LoRA worse (0.110), and rank 32 recovered about half of the gap (0.127 → 0.136, full 0.145). The evidence leans toward **capacity** (your "smaller manifold" explanation) rather than under-training, but the rank-32 gain is below the 0.01 single-seed noise level, so it is not conclusive.

## Code changes for these runs

| File | Change |
|---|---|
| `water-debris-seg/scripts/train_yolo_lars.py` | `--model` (pretrained weights), `--lr0` (explicit AdamW lr, otherwise `auto`), `--tag` (run naming); recorded in `run_spec.json` |
| `water-debris-seg/scripts/snellius/ablation10_yolo_lars.sbatch` | `TAG` and `EXTRA` (extra training arguments) |

## Reproduce

```bash
cd water-debris-seg
S=scripts/snellius/ablation10_yolo_lars.sbatch
sbatch --export=ALL,VARIANT=lora,STAGE=A,TAG=lora_lr10,EXTRA=--lr0=0.00769 $S
sbatch --export=ALL,VARIANT=lora,STAGE=A,TAG=lora_r32,LORA_R=32,LORA_ALPHA=64 $S
sbatch --export=ALL,VARIANT=full,STAGE=A,TAG=full_11s,EXTRA=--model=yolo11s.pt $S
sbatch --export=ALL,VARIANT=full,STAGE=A,LOSS=focal,TAG=full_focal_a075,EXTRA=--focal-alpha=0.75 $S
# stage B: the same exports with STAGE=B and --dependency=afterok:<its stage-A job>
```
