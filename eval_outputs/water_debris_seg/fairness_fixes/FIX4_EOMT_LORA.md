# Fix 4 — EoMT: full fine-tuning vs LoRA (EoMT only)

> **Results (2026-10-26):** all runs of this fix are done; values with links to their JSON are in [RESULTS.md](RESULTS.md). ⏳ cells below are the plan as written before the runs.

**Status: plan, not started (decision 2026-10-23).** Back to the summary: [README.md](README.md). LoRA is compared for **EoMT only**; YOLO LoRA is discarded (ablation 10's YOLO LoRA runs stay as references, not in the comparison).

## Motivation

EoMT is a ViT-B (DINOv2) encoder (~86M parameters) with 200 queries in its last 3 blocks, against YOLOv11s's 9.4M; the exact EoMT count is logged by the first run. All EoMT fine-tuning so far updated every weight (ablation 03). LoRA trains a small low-rank update on the frozen pretrained weights. The question: does EoMT need full fine-tuning on these small datasets (LaRS 2,605 train images; FloW 1,200; CANSURF 4,950), or does LoRA match it with far fewer trainable parameters and less risk of overfitting?

## What is trained

| Arm | Trainable | Frozen |
|---|---|---|
| **Full fine-tuning** | all weights (as ablation 03) | — |
| **LoRA** | LoRA factors on every encoder block's attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) linear layers, rank r = 16, α = 32; plus the queries, class and mask heads and LayerNorms (new or task-specific) | the pretrained encoder weights |

LoRA for linear layers is a direct analogue of `water_debris_seg/yolo_lora.py` (conv layers): frozen weight as a buffer, zero-initialised B, merge-able for inference. EoMT's optimizer applies layer-wise learning-rate decay by parameter name; the LoRA factors must receive their block's learning rate (checked by a unit test before training).

## Where

| Dataset | Full | LoRA | Main-table rows |
|---|---|---|---|
| LaRS (panoptic, 11 classes) | ⏳ rerun with the held-out protocol (ablation 03's 0.395 used a fixed 12-epoch recipe) | ⏳ | EoMT panoptic (full) and EoMT panoptic LoRA |
| FloW (SAM masks, fix 3) | ⏳ (fix 3) | ⏳ | EoMT (SAM masks) and EoMT LoRA (SAM masks) |
| CANSURF (SAM masks, fix 3) | ⏳ (fix 3) | ⏳ | same |

Both arms use the same protocol: stage A on the training split minus a held-out slice picks the epoch count; stage B retrains on the full training split and keeps the last checkpoint; the test split is scored once. Learning rate: EoMT's default for full fine-tuning; for LoRA, the same plus a ×10 variant in stage A only if LoRA under-trains (the YOLO follow-up showed a global ×10 rate hurts, so LoRA-only learning rates would be used).

## Attention-mask annealing (affects both arms equally)

EoMT anneals masked attention off during training so that inference runs without masks. Ablation 03 used COCO's schedule (annealing starts at step 14,782) for a 7,812-step run, so **masked attention stayed fully on** in training and inference. This is consistent (no train/test mismatch) but is not EoMT's intended mask-free inference. For fix 4 (and fix 3), **both arms scale the annealing schedule to the run length** (same relative start and end points as EoMT's COCO recipe). The rerun of full fine-tuning on LaRS also shows whether this changes EoMT's PQ against ablation 03.

## Expected outputs

| Output | Path / W&B |
|---|---|
| LoRA module for ViT linear layers + tests | `water-debris-seg/water_debris_seg/eomt_lora.py`, `water-debris-seg/tests/test_eomt_lora.py` |
| Configs | `water-debris-seg/configs/fix4_eomt_{full,lora}_{lars,flow,cansurf}_stage{A,B}.yaml` |
| Runs | ⏳ W&B group `fix4_eomt_lora` |
| Results | LaRS: PQ, mIoU, box AP (as fix 1); FloW / CANSURF: box AP debris-only + class-agnostic (as fix 3); trainable-parameter counts, training time, inference speed |
| Results page | `fairness_fixes/FIX4_RESULTS.md` |

## Cost

| Run | Basis | GPU time |
|---|---|---|
| LaRS full + LoRA, stage A + B each | ablation 03: 41 min for 12 epochs on one A100; stage A needs a longer budget | ~2 × 1.5–2 h |
| FloW full + LoRA | scaled from LaRS (×0.46) | ~2 × 1–1.5 h (full arm is part of fix 3) |
| CANSURF full + LoRA | scaled from LaRS (×1.9) | ~2 × 4–6 h (full arm is part of fix 3) |
| **Added by fix 4** (LoRA arms + LaRS full rerun) | | **~8–11 GPU h**; parallel with fix 3, so little extra wall-clock time |

## Caveats

- Single seed per arm; differences under ~0.01 are not conclusive (as in ablation 10).
- LoRA rank and target layers are one setting (r = 16 on attention + MLP); a rank sweep is not planned.
- Annealing is scaled for both arms, so the full LaRS rerun is not identical to ablation 03; ablation 03 stays as a reference.
