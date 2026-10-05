# Ablation 04 — FloW-Img mask→box mAP (eval only)

**Question.** As a **detector**, how does EoMT/VidEoMT (zero-shot vs LaRS-trained) perform on **inland floating waste** boxes?

**Expected outcome.** Low mAP vs FloW paper detectors; **AP_S** especially poor (many objects &lt;32²). Validates need for domain-specific training or YOLO.

**Hard rule.** **No training** on FloW. Inference + mask→box only.

## Setup

| Item | Value |
|---|---|
| Data | FloW-Img 800 test (or official split) |
| Weights | 01 zero-shot and/or 03 LaRS panoptic |
| Metrics | bbox mAP, AP_S |

## Results

| Weights | mAP | AP_S | Source |
|---|---|---|---|
| EoMT zero-shot | — | — | *pending* |
| EoMT LaRS-FT | — | — | *pending* |

## Verdict

*Pending.* If FloW download blocked → **not run** (decision log).

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- FloW adapter (future)
- mask→box eval (shared)
