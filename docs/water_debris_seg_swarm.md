# Water debris segmentation — agent swarm

**Status:** Stage 0 scaffold complete. Stage 1+ gated on disk slack and GPU inventory.  
**Spec:** [`water_debris_seg_benchmark.md`](water_debris_seg_benchmark.md)  
**Decision log:** [`water_debris_seg_decision_log.md`](water_debris_seg_decision_log.md)  
**Artifacts:** [`eval_outputs/water_debris_seg/`](../eval_outputs/water_debris_seg/) (repo root)

**Research question:** Can general-purpose mask transformers (EoMT, VidEoMT) match or beat specialized detectors on nearshore debris, and does temporal modeling help on USV video?

---

## Gate rule

No dataset downloads, clones, or GPU jobs until:

1. [`data/water_debris/DISK_BUDGET.md`](../data/water_debris/DISK_BUDGET.md) shows free space ≥ next stage + 20% slack.
2. [`eval_outputs/water_debris_seg/GPU_INVENTORY.md`](../eval_outputs/water_debris_seg/GPU_INVENTORY.md) lists ≥1 working CUDA device (excluding `elec-gpue002` on Slurm).
3. Ablation README stubs exist under `eval_outputs/water_debris_seg/ablations/`.

---

## Agent swarm

```mermaid
flowchart TD
  subgraph wave0 [Wave0_Foundation]
    A0[Agent0_DocsScaffold]
    A1[Agent1_DiskGate]
    A2[Agent2_GPUProbe]
  end
  subgraph wave1 [Wave1_Data]
    B1[Agent3_LaRS]
    B2[Agent4_CANSURF]
    B3[Agent5_FloW]
    B4[Agent6_WSODD]
    B5[Agent7_TaxonomyAdapters]
  end
  A0 --> A1
  A0 --> A2
  A1 --> B1
  A2 --> B1
  B1 --> B5
  B2 --> B5
  B3 --> B5
  B4 --> B5
  subgraph wave2 [Wave2_Smoke]
    C1[Agent8_EoMT_smoke]
    C2[Agent9_VidEoMT_smoke]
    C3[Agent10_YOLO_smoke]
  end
  B5 --> C1
  B5 --> C2
  B5 --> C3
  subgraph wave3 [Wave3_ZeroshotEval]
    D1[Agent11_EoMT_LaRS]
    D2[Agent12_mask2box_all]
  end
  C1 --> D1
  C1 --> D2
  subgraph wave4 [Wave4_LaRS_Finetune]
    E1[Agent13_LaRS_semantic]
    E2[Agent14_LaRS_panoptic]
  end
  D1 --> E1
  E1 --> E2
  subgraph wave5 [Wave5_YOLO]
    F1[Agent15_YOLO_CANSURF]
    F2[Agent16_YOLO_FloW_WSODD]
  end
  C3 --> F1
  C3 --> F2
  subgraph wave6 [Wave6_VidEoMT]
    G1[Agent17_VidEoMT_LaRS]
  end
  C2 --> G1
  E2 --> G1
  subgraph wave7 [Wave7_Report]
    H1[Agent18_CrossDataset]
    H2[Agent19_REPORT_plots]
  end
  E2 --> H1
  F1 --> H1
  G1 --> H1
  H1 --> H2
```

---

## Timeline

| Stage | Days | Owner wave | Gate to next |
|---|---|---|---|
| **0** Foundation | 0.5 | Wave 0 | Spec + stubs; disk + GPU docs |
| **1** Data + adapters | 1–3 | Wave 1 | File manifest; skip blocked sets honestly |
| **2** Smoke | 0.5–1 | Wave 2 | Overlays + 1 metric row per stack |
| **3** Zero-shot eval | 1–2 | Wave 3 | Ablations 01, 04–06 (no `fit` on box sets) |
| **4** LaRS fine-tune | 2–4 | Wave 4 | Ablations 02–03; last ckpt only |
| **5** YOLO specialized | 1–2 | Wave 5 | Ablation 08 |
| **6** VidEoMT | 2–3 | Wave 6 | Ablation 07; LaRS sequences on disk |
| **7** Report | 1 | Wave 7 | `REPORT.md`, `metrics.json`, plots |

**~10–16 working days** with LaRS + CANSURF on day 1.

---

## Agent prompts (copy-paste)

### Agent 0 — Docs scaffold

```
Owns: docs/water_debris_seg_*.md, eval_outputs/water_debris_seg/ tree
Tasks: benchmark spec, swarm doc, ablation README stubs, REPORT catalog
Depends: nothing
Checklist:
  [ ] Hard rule visible at top of benchmark.md
  [ ] Each ablation has Question + expected outcome
  [ ] decision log baseline entry
```

### Agent 1 — Disk gate

```
Owns: data/water_debris/DISK_BUDGET.md
Tasks: df -h before each stage; compare to plan table + 20% slack
Depends: Agent 0
Checklist:
  [ ] Current avail bytes recorded
  [ ] Next stage GO/NO-GO with reason
```

### Agent 2 — GPU probe

```
Owns: eval_outputs/water_debris_seg/GPU_INVENTORY.md
Tasks: nvidia-smi -L; per-device torch alloc; mark elec-gpue002 excluded on Slurm
Depends: Agent 0
Checklist:
  [ ] working_devices[] listed
  [ ] failed_devices[] with error text
```

### Agent 3 — LaRS download

```
Owns: data/water_debris/lars/
Tasks: form download images + annotations; optional sequences later
Depends: Agents 1, 2 (GO on disk)
Checklist:
  [ ] panoptic_annotations.json + masks present
  [ ] manifest sha256 in sources/
```

### Agent 4 — CANSURF

```
Owns: data/water_debris/cansurf/
Tasks: clone/GitHub or Zenodo raw train/val only (no 57k aug expansion)
Depends: Disk gate
```

### Agent 5 — FloW

```
Owns: data/water_debris/flow/
Tasks: ORCA application download FloW-Img; log if blocked
Depends: Disk gate
```

### Agent 6 — WSODD

```
Owns: data/water_debris/wsodd/
Tasks: Baidu download; VOC2COCO if needed
Depends: Disk gate (download last — largest)
```

### Agent 7 — Taxonomy adapters

```
Owns: water-debris-seg/datasets/ (future) or third_party adapters
Tasks: unified class map; LaRS→COCO panoptic for EoMT; boxes→COCO for mAP eval
Depends: Agents 3–6 (partial OK)
Tests: import smoke + 1 batch load per available dataset
```

### Agent 8 — EoMT smoke

```
Owns: third_party/eomt/, eval_outputs/.../smoke/eomt/
Tasks: conda env; 8–16 LaRS or COCO val images; overlay PNGs
Depends: Agent 7 (or COCO-only fallback)
Hard rule: no fit on FloW/CANSURF/WSODD
```

### Agent 9 — VidEoMT smoke

```
Owns: third_party/videomt/, smoke/videomt/
Tasks: detectron2 install; eval-only 1 clip if LaRS sequences exist
Depends: Agent 8 optional parallel
```

### Agent 10 — YOLO smoke

```
Owns: smoke/yolo/
Tasks: ultralytics; 1-epoch micro-train on CANSURF subset OR dry-run predict
Depends: Agent 4
```

### Agent 11 — EoMT zero-shot LaRS

```
Owns: ablations/01_zeroshot_lars/
Tasks: COCO/ADE ckpt validate on LaRS val; PQ/mIoU table
Depends: Agent 8
```

### Agent 12 — mask→box eval

```
Owns: ablations/04_flow_mask2box, 05_cansurf_small_object, 06_wsodd_distractors
Tasks: shared script masks_to_coco_boxes.py + pycocotools eval; eval-only weights
Depends: Agents 8, 7
Hard rule: NO training on these datasets
Parallel: 1 GPU per dataset job
```

### Agent 13–14 — LaRS fine-tune

```
Owns: ablations/02_finetune_lars_semantic, 03_finetune_lars_panoptic
Tasks: eomt main.py fit; load_ckpt_class_head False; DDP batch 16
Depends: Agent 11
Hard rule: LaRS ONLY
```

### Agent 15–16 — YOLO trains

```
Owns: ablation/08_specialized_yolo/
Tasks: YOLOv11 on CANSURF; optional FloW/WSODD-rubbish
Depends: Agent 10
```

### Agent 17 — VidEoMT LaRS

```
Owns: ablation/07_videomt_temporal/
Tasks: download LaRS sequences; train or eval-only; compare to EoMT still
Depends: Agents 13–14, LaRS sequences on disk
```

### Agent 18–19 — Report

```
Owns: REPORT.md, metrics.json, plots/
Tasks: cross-dataset table; copy sources/; experiment catalog
Depends: all completed ablations
Checklist:
  [ ] Missing datasets marked "not run" in README + decision log
  [ ] No invented metrics
```

---

## Parallelism and Slurm

- **Eval:** `#SBATCH --gres=gpu:1` per (model × dataset); array jobs OK.
- **EoMT fit:** `--gres=gpu:4` when 4 healthy devices; else accumulate to batch 16.
- **Always:** `#SBATCH --exclude=elec-gpue002` on this cluster.
- Set `CUDA_VISIBLE_DEVICES` from `GPU_INVENTORY.md` working list only.

---

## Logging

| File | Purpose |
|---|---|
| `data/water_debris/DISK_BUDGET.md` | Per-stage disk gates |
| `eval_outputs/water_debris_seg/GPU_INVENTORY.md` | Device probe |
| `docs/water_debris_seg_decision_log.md` | Human-readable decisions |
| `eval_outputs/water_debris_seg/ablations/*/sources/jobids.json` | Slurm IDs (when used) |

---

## Definition of done (benchmark)

- Headline table in `REPORT.md`: EoMT / VidEoMT / YOLO × LaRS + available box sets.
- Every number has a **Source** path in its ablation README.
- Hard rule satisfied: no EoMT/VidEoMT checkpoint trained on FloW/CANSURF/WSODD.
