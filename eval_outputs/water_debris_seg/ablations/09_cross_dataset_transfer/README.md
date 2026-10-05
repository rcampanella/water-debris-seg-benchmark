# Ablation 09 — Cross-dataset transfer (LaRS-trained → box sets)

**Question.** After **LaRS-only** fine-tuning, does maritime panoptic knowledge transfer to **FloW / CANSURF / WSODD** via mask→box mAP?

**Expected outcome.** Better than zero-shot on semantic “obstacle” alignment; still likely **below** YOLO trained on target domain (09 vs 08).

**Hard rule.** Checkpoint from ablations 02–03 only; **eval only** on box sets.

## Setup

| Checkpoint | Eval on |
|---|---|
| EoMT LaRS semantic (02) | mask→box on all three box sets |
| EoMT LaRS panoptic (03) | same |

## Results

| Ckpt | FloW mAP | CANSURF mAP | WSODD rubbish AP | Source |
|---|---|---|---|---|
| LaRS semantic | — | — | — | *pending* |
| LaRS panoptic | — | — | — | *pending* |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| ablations/02/03 ckpt paths | sources/ | — |

## Code components

- Same mask→box pipeline as ablations 04–06
