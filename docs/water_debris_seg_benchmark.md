# Water-surface debris segmentation benchmark

Nearshore USV camera perception: segment and detect **objects on the water surface** (debris pickup vs obstacles). Companion execution plan: [`water_debris_seg_swarm.md`](water_debris_seg_swarm.md). Decision log: [`water_debris_seg_decision_log.md`](water_debris_seg_decision_log.md).

Artifacts live under [`eval_outputs/water_debris_seg/`](../eval_outputs/water_debris_seg/).

---

## Hard rule

**Train EoMT and VidEoMT masks only on LaRS. On FloW, CANSURF, and WSODD, do not train those models. Evaluate them there as detectors: masks → axis-aligned boxes → COCO mAP / AP_S.**

| | LaRS | FloW / CANSURF / WSODD |
|---|---|---|
| Annotation | Pixel masks (semantic + panoptic) | Boxes only |
| EoMT / VidEoMT **train** | Yes (fine-tune class head off COCO/ADE / video-zoo weights) | **No** — not even with rectangle- or SAM-filled “masks” |
| EoMT / VidEoMT **eval** | PQ, mIoU (and VidEoMT temporal metrics) | **mask→box mAP only** |
| YOLO **train + eval** | Optional (boxes from LaRS things) | Yes — boxes are native GT |

“Fine-tune” in this benchmark always means **LaRS**, never FloW/CANSURF/WSODD. Zero-shot on those three means: official COCO/ADE (or LaRS-trained) weights, then mask→box. It does **not** mean training on their train splits.

EoMT/VidEoMT losses need dense GT masks. Boxes are loose bounds on water (glare, ripples, partial submersion). Rectangle or SAM-filled boxes as mask GT would train “paint the rectangle,” not segment debris.

---

## Mission and operational taxonomy

| Unified class | Meaning | Dataset hooks |
|---|---|---|
| **debris** | Floating waste, rubbish, cans, float | FloW waste, CANSURF can, WSODD rubbish, LaRS Float/Other |
| **dynamic_obstacle** | Boats, buoys, swimmers, animals | LaRS things; WSODD boat/ship/buoy/person/animal |
| **static_context** | Shore, rocks, bridges, harbor | WSODD bridge/harbor/tree/grass |
| **water / sky** | Stuff regions | LaRS semantic IDs 1 (water), 2 (sky) |

Headline: **surface debris**. Secondary: avoid confusing debris with buoys/boats. Tertiary: temporal stability (VidEoMT vs per-frame EoMT).

---

## Datasets

| Dataset | Link | Scale | Labels |
|---|---|---|---|
| **LaRS** | [Project page](https://lojzezust.github.io/lars-dataset) | 4000+ key frames; 9 prev frames (~40k); images ~966 MB, ann ~22 MB, sequences ~13.6 GB | Panoptic COCO + semantic PNG + 19 scene attributes |
| **FloW-Img** | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.html) | 2000 imgs, 5271 boxes; 1200/800 split; many &lt;32² | Boxes; 200 unlabeled videos |
| **CANSURF** | [arXiv:2605.16774](https://arxiv.org/abs/2605.16774), [GitHub](https://github.com/ZaidAljundiHW2/CANSURF) | ~7.2k raw YOLO; paper ~57k aug (do not materialize full aug on disk) | 1 class: can |
| **WSODD** | [GitHub](https://github.com/sunjiaen/WSODD) | 7467 × 1920×1080, 14 classes, 21911 instances | VOC/COCO boxes; rubbish = debris |

LaRS panoptic categories (ID → name): Static Obstacle (1), Water (3), Sky (5); things: Boat/ship (11), Row boats (12), Paddle board (13), Buoy (14), Swimmer (15), Animal (16), Float (17), Other (19).

---

## Models

| Model | Role | References |
|---|---|---|
| **EoMT** | General-purpose image segmentation | [Paper](https://arxiv.org/pdf/2503.19108), [Code](https://github.com/tue-mps/eomt) |
| **VidEoMT** | General-purpose online video segmentation | [Paper](https://arxiv.org/abs/2602.17807), [Code](https://github.com/tue-mps/videomt) |
| **YOLOv11** | Specialized detection baseline | CANSURF paper recipe; optional SAHI |

Official EoMT/VidEoMT weights are pre-trained on COCO / ADE / video benchmarks (not maritime). **Our** mask training runs only on LaRS.

---

## Evaluation protocols

### LaRS (native segmentation)

- **Semantic:** mIoU on obstacle / water / sky (ignore 255).
- **Panoptic:** PQ, SQ, RQ; optional slices on `image_annotations.json` attributes (glare, waves, weather).
- **VidEoMT:** same on keyframes + temporal consistency (mask flicker, thing ID stability over 9-frame context).

### FloW / CANSURF / WSODD (detection bridge)

Same for EoMT and VidEoMT (VidEoMT may run online; scored object is per-frame or last-frame mask):

1. Run published or **LaRS-fine-tuned** checkpoint (no weight updates on these sets).
2. Filter masks by score threshold (record in ablation README).
3. Map predicted classes → dataset labels (document drops/distractors).
4. Mask → axis-aligned box (min/max of foreground pixels).
5. **COCO bbox mAP** and **AP_S** (area &lt; 32²).
6. JSONL per-image predictions + summary table with **Source** column.

Optional diagnostic only: `box_proxy_IoU` vs GT rectangle — never headline PQ/mIoU.

### YOLO

Train and eval with native box GT on CANSURF (+ FloW / WSODD-rubbish if downloaded). Compare to EoMT mask→box mAP on the same splits.

---

## EoMT eval ladder

1. **Smoke** — 8–16 images, COCO/ADE checkpoint, overlays.
2. **Zero-shot (eval only)** — LaRS PQ/mIoU + FloW/CANSURF/WSODD mask→box mAP.
3. **Fine-tune (LaRS only)** — semantic 3-class + panoptic 11-class; `load_ckpt_class_head False`; batch 16 total (e.g. 4×4 GPUs).
4. **Cross-dataset (eval only)** — LaRS checkpoint → mask→box on box sets.

## VidEoMT eval ladder

- Train/fine-tune on **LaRS sequences only** when training is enabled.
- LaRS: online vs EoMT still; PQ/mIoU + temporal metrics.
- Box sets: mask→box mAP on stills only (WSODD: no temporal claim).
- FPS/GFLOPs via upstream `benchmark.py`.

---

## Experiment layout (OpenFly convention)

Each ablation: **Question → Motivation / expected outcome → Setup → Results (Source column) → Verdict → Resources copied here → Code components**.

```
eval_outputs/water_debris_seg/
  REPORT.md              # catalog + headline tables
  ablations/01_zeroshot_lars/ ...
  smoke/  benchmark/  plots/  sources/
```

See [`eval_outputs/water_debris_seg/REPORT.md`](../eval_outputs/water_debris_seg/REPORT.md) for the experiment catalog.

---

## Disk budget (gate before every stage)

Abort if free space &lt; stage need + **20% slack**. Record each gate in [`data/water_debris/DISK_BUDGET.md`](../data/water_debris/DISK_BUDGET.md).

| Stage | Increment | Notes |
|---|---|---|
| Code + ViT weights | 8–15 GB | EoMT + VidEoMT repos + B/L ckpts |
| LaRS stills + ann | ~1 GB | Required first |
| LaRS sequences | ~14 GB | Defer until VidEoMT |
| CANSURF raw | 3–8 GB | No 57k aug materialization |
| FloW-Img | 5–15 GB | Videos optional |
| WSODD | 10–25 GB | Download last |
| Train/eval artifacts | 20–40 GB | Keep last ckpt |

Full working set ~80–120 GB. Minimum path ~15–25 GB (LaRS stills + CANSURF raw + EoMT-B).

---

## GPU rules

Before CUDA jobs: update [`eval_outputs/water_debris_seg/GPU_INVENTORY.md`](../eval_outputs/water_debris_seg/GPU_INVENTORY.md).

- Per-device `nvidia-smi -L` + torch alloc test.
- Slurm: **`--exclude=elec-gpue002`** (cluster convention in this workspace).
- Drop GPUs that fail init or ECC errors.
- Parallel eval: 1 GPU × (model × dataset). No two ViT-L fine-tunes on one GPU.

---

## Out of scope

- Closed-loop boat / OpenFly sim
- FloW-RI radar fusion
- Training EoMT/VidEoMT on FloW, CANSURF, WSODD
- Headline mask IoU vs filled boxes
- Weakly-supervised mask fine-tune on box sets (unless a future named ablation)

---

## Citations

LaRS: Žust et al., ICCV 2023. FloW: Cheng et al., ICCV 2021. CANSURF: Aljundi et al., ICSPIS 2025. WSODD: Frontiers in Neurorobotics 2021. EoMT: Kerssies et al., CVPR 2025. VidEoMT: Norouzi et al., CVPR 2026.
