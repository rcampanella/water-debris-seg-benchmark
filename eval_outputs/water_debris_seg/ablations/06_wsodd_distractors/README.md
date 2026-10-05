# Ablation 06 — WSODD rubbish vs distractors (eval only)

**Question.** On a **14-class harbor** dataset, does a segmenter-as-detector find **rubbish** without confusing **boats, buoys, balls, animals**?

**Expected outcome.** High precision on rubbish requires LaRS-FT or YOLO; zero-shot EoMT likely fires on generic “object” classes → FP on boats.

**Hard rule.** **No training** on WSODD for EoMT/VidEoMT. Per-frame mask→box only (no VidEoMT temporal claim).

## Setup

| Item | Value |
|---|---|
| Target class | rubbish (669 instances in dataset) |
| Metrics | mAP on rubbish; FP rate on boat/buoy |
| Resolution | 1920×1080 — record FPS |

## Results

| Weights | rubbish AP | boat/buoy FP notes | Source |
|---|---|---|---|
| EoMT zero-shot | — | — | *pending* |
| EoMT LaRS-FT | — | — | *pending* |

## Verdict

*Pending.* Baidu download blocked → **not run**.

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- WSODD VOC2COCO ([WSODD repo](https://github.com/sunjiaen/WSODD))
