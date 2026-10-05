# Ablation 05 — CANSURF mask→box mAP (eval only)

**Question.** Can mask transformers detect **reflective aluminum cans** at distance compared to YOLOv11 (and optional SAHI)?

**Expected outcome.** YOLO **beats** EoMT on mAP/recall; EoMT may miss far-field AP_S unless SAHI-style tiling at inference (optional, document separately).

**Hard rule.** **No training** on CANSURF for EoMT/VidEoMT.

## Setup

| Item | Value |
|---|---|
| Data | CANSURF raw val (~1.7k) |
| Compare | Ablation 08 YOLO on same split |
| Metrics | mAP, AP_S, far-field bin (distance proxy via box area) |

## Results

| Method | mAP | AP_S | Source |
|---|---|---|---|
| EoMT zero-shot | — | — | *pending* |
| EoMT LaRS-FT | — | — | *pending* |
| YOLOv11 | — | — | ablations/08 |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- [CANSURF repo](https://github.com/ZaidAljundiHW2/CANSURF) layout reference
