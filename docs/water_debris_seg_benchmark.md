# Water-surface debris segmentation benchmark

Nearshore USV camera perception: segment and detect **objects on the water surface** (debris pickup vs obstacles). Companion execution plan: [`water_debris_seg_swarm.md`](water_debris_seg_swarm.md). Decision log: [`water_debris_seg_decision_log.md`](water_debris_seg_decision_log.md).

Artifacts live under [`eval_outputs/water_debris_seg/`](../eval_outputs/water_debris_seg/).

---

## Training rule (revised 2026-10-23)

**Both model families are trained on each dataset they are compared on, with the same held-out protocol, and scored on the same ground truth.** Plan, target table and decisions: [`fairness_fixes/README.md`](../eval_outputs/water_debris_seg/fairness_fixes/README.md).

| | LaRS | FloW / CANSURF |
|---|---|---|
| Annotation | Pixel masks (semantic + panoptic) | Boxes only |
| EoMT **train** | Yes, panoptic (semantic fine-tune kept as a reference only) | Yes, on **SAM masks prompted by the GT boxes** (not filled rectangles), after a mask quality check |
| EoMT **eval** | PQ, mIoU, box AP (masks → min/max boxes) | box AP (masks → min/max boxes) |
| YOLOv11s **train + eval** | Boxes traced from LaRS masks; box AP; PQ / mIoU via YOLOv11s + SAM | Native boxes; box AP |
| VidEoMT | LaRS sequences only; separate temporal table | not evaluated |

On FloW / CANSURF the main number counts **debris classes only** (class-agnostic as reference). Superseded rule (2026-10-05 to 2026-10-23): EoMT/VidEoMT masks trained only on LaRS, FloW / CANSURF eval-only for them, because box GT cannot train segmentation. Box-prompted SAM masks answer that concern better than filled rectangles; the SAM quality check and a SAM-box ceiling quantify what remains.

**WSODD** ([GitHub](https://github.com/sunjiaen/WSODD)) is **out of scope for v1** (Baidu-only hosting). Ablation 06 is cancelled; see decision log 2026-10-07.

EoMT losses need dense GT masks. Boxes are loose bounds on water (glare, ripples, partial submersion), so filled rectangles would train "paint the rectangle". Fix 3 therefore uses SAM masks prompted by each box and checks their quality before training.

---

## Mission and operational taxonomy

| Unified class | Meaning | Dataset hooks |
|---|---|---|
| **debris** | Floating waste, rubbish, cans, float | FloW waste, CANSURF can, LaRS Float/Other |
| **dynamic_obstacle** | Boats, buoys, swimmers, animals | LaRS panoptic things (Boat, Buoy, Swimmer, …) |
| **static_context** | Shore, rocks, bridges, harbor | LaRS static obstacle / scene context (semantic) |
| **water / sky** | Stuff regions | LaRS semantic IDs 1 (water), 2 (sky) |

Headline: **surface debris**. Secondary: avoid confusing debris with buoys/boats. Tertiary: temporal stability (VidEoMT vs per-frame EoMT).

---

## Datasets

| Dataset | Link | Scale | Labels |
|---|---|---|---|
| **LaRS** | [Project page](https://lojzezust.github.io/lars-dataset) | 4000+ key frames; 9 prev frames (~40k); images ~966 MB, ann ~22 MB, sequences ~13.6 GB | Panoptic COCO + semantic PNG + 19 scene attributes |
| **FloW-Img** | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.html) | 2000 imgs, 5271 boxes; 1200/800 split; many &lt;32² | Boxes; 200 unlabeled videos |
| **CANSURF** | [arXiv:2605.16774](https://arxiv.org/abs/2605.16774), [GitHub](https://github.com/ZaidAljundiHW2/CANSURF) | ~7.2k raw YOLO; paper ~57k aug (do not materialize full aug on disk) | 1 class: can |
LaRS panoptic categories (ID → name): Static Obstacle (1), Water (3), Sky (5); things: Boat/ship (11), Row boats (12), Paddle board (13), Buoy (14), Swimmer (15), Animal (16), Float (17), Other (19).

---

## Models

| Model | Role | References |
|---|---|---|
| **EoMT** | General-purpose image segmentation | [Paper](https://arxiv.org/pdf/2503.19108), [Code](https://github.com/tue-mps/eomt) |
| **VidEoMT** | General-purpose online video segmentation | [Paper](https://arxiv.org/abs/2602.17807), [Code](https://github.com/tue-mps/videomt) |
| **YOLOv11s** | Specialized detection baseline (YOLOv11n results kept as reference) | CANSURF paper's model ([`train.py`](https://github.com/ZaidAljundiHW2/CANSURF): `yolo11s.pt`, imgsz 640, batch 64, 150 epochs); our runs use the held-out protocol instead of the paper's early stopping. Ablation 08 used YOLOv11n with 50 epochs and batch 16, not the paper's recipe |
| **SAM** (frozen) | Box → mask | YOLOv11s + SAM on LaRS; EoMT + SAM control; pseudo-masks for EoMT on FloW / CANSURF |

Official EoMT/VidEoMT weights are pre-trained on COCO / ADE / video benchmarks (not maritime). Our mask training runs on LaRS, and (fix 3) on FloW / CANSURF SAM masks.

---

## Evaluation protocols

### LaRS (native segmentation)

- **Semantic:** mIoU on obstacle / water / sky (ignore 255).
- **Panoptic:** PQ, SQ, RQ; optional slices on `image_annotations.json` attributes (glare, waves, weather).
- **VidEoMT:** same on keyframes + temporal consistency (mask flicker, thing ID stability over 9-frame context).

### FloW / CANSURF (detection)

For EoMT (VidEoMT is not evaluated here):

1. Run the zero-shot, LaRS-trained (cross-domain) or dataset-trained (fix 3) checkpoint.
2. Keep EoMT's instance detections with their confidence (class probability × mean mask probability).
3. Count only the **debris classes** (main number); report a **class-agnostic** score as reference; keep each prediction's original class.
4. Mask → axis-aligned box (min/max of foreground pixels).
5. **COCO bbox mAP** and **AP_S** (area &lt; 32²).
6. JSONL per-image predictions + summary table with **Source** column.

Optional diagnostic only: `box_proxy_IoU` vs GT rectangle — never headline PQ/mIoU.

### YOLO

YOLOv11s: train and eval with native box GT on FloW and CANSURF, and on boxes traced from the LaRS masks (held-out protocol everywhere). Compare to EoMT on the same splits and ground truth.

---

## EoMT eval ladder

1. **Smoke** — 8–16 images, COCO/ADE checkpoint, overlays.
2. **Zero-shot (eval only)** — LaRS PQ/mIoU + FloW/CANSURF mask→box mAP.
3. **Fine-tune (LaRS only)** — semantic 3-class + panoptic 11-class; `load_ckpt_class_head False`; batch 16 total (e.g. 4×4 GPUs).
4. **Cross-dataset (eval only)** — LaRS checkpoint → mask→box on box sets.

## VidEoMT eval ladder

- Train/fine-tune on **LaRS sequences only** when training is enabled.
- LaRS: online vs EoMT still; PQ/mIoU + temporal metrics.
- Box sets: mask→box mAP on stills only.
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
| Train/eval artifacts | 20–40 GB | Keep last ckpt |

Full working set ~80–120 GB. Minimum path ~15–25 GB (LaRS stills + CANSURF raw + EoMT-B).

---

## GPU rules (Snellius)

Before CUDA jobs: update [`eval_outputs/water_debris_seg/GPU_INVENTORY.md`](../eval_outputs/water_debris_seg/GPU_INVENTORY.md).

- Probe on **`gpu_a100` / `gpu_h100` compute nodes** via `srun` or `sbatch` (login nodes have no GPU).
- Per-device `nvidia-smi -L` + torch CUDA alloc test in the benchmark environment.
- Drop nodes/GPUs that fail init; skip Slurm nodes in `drained` / `down` when possible.
- Parallel eval: 1 GPU × (model × dataset). No two ViT-L fine-tunes on one GPU.

## Storage rules (Snellius)

See [`data/water_debris/DISK_BUDGET.md`](../data/water_debris/DISK_BUDGET.md).

- **Home** (`/home/$USER`, 200 GiB default): git repo, manifests, reports, final checkpoints — **persistent**, nightly backup.
- **Scratch-shared**: datasets and training I/O — **not permanent** (14-day inactivity deletion); full benchmark working set lives here.
- Do not run heavy parallel I/O from home; sync precious outputs off scratch after each job.

---

## Out of scope

- Closed-loop boat / OpenFly sim
- FloW-RI radar fusion
- Training VidEoMT on FloW, CANSURF (EoMT is trained there on SAM masks since 2026-10-23)
- **WSODD** dataset and ablation 06 (Baidu access; see decision log 2026-10-07)
- Headline mask IoU vs filled boxes
- Weakly-supervised mask fine-tune on box sets (unless a future named ablation)

---

## Citations

LaRS: Žust et al., ICCV 2023. FloW: Cheng et al., ICCV 2021. CANSURF: Aljundi et al., ICSPIS 2025. EoMT: Kerssies et al., CVPR 2025. VidEoMT: Norouzi et al., CVPR 2026.
