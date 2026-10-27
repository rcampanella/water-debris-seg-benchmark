# Focal loss vs BCE on LaRS: loss-behaviour and qualitative study (plan, for approval)

**Status: plan only, nothing run yet.** After approval, results go to a separate report, `FOCAL_STUDY.md` (written for reuse in the qualitative presentation), with figures and numbers in `focal_study/`.

**Why.** Ablation 10b showed focal loss did not beat BCE ([README.md § 10b](README.md#ablation-10b--focal-loss)): full object mAP 0.145 → 0.129, small and rare classes down, only Static Obstacle up. This study looks inside the loss to see *why*: which samples each loss pays attention to on our data, and what that does to the detections.

**Reference.** S. Trivedi, *Understanding Focal Loss — A Quick Read* (VisionWizard, Medium), which summarises Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet, ICCV 2017). The study follows its vocabulary and tests its main claims on our models and data.

---

## 1. Claims from the post that the study tests

| # | Claim in the post | How we test it on LaRS |
|---|---|---|
| C1 | One-stage detectors face a huge **background ≫ foreground** imbalance | Count positive vs negative samples per image in YOLOv11 (§ 4.1) |
| C2 | Most samples are **easy negatives** that carry no useful signal, and their **summed loss overwhelms** the total | Share of BCE loss **and gradient** coming from easy negatives (§ 4.2) |
| C3 | Focal loss down-weights easy samples far more than hard ones (post: CE/FL ratio 360 for an easy positive vs 40 for a hard one) | Same numbers recomputed for our γ, α (§ 4.0), then measured on real predictions (§ 4.2) |
| C4 | There is also a **foreground–foreground** (class) imbalance, which α and γ do not target directly | Positives per class, e.g. Boat/ship vs Float (§ 4.1) |

## 2. Adapting the post to YOLOv11 (important differences)

The post describes anchor-based detectors (YOLOv3, RetinaNet). YOLOv11 differs in ways that change what "positive", "easy" and "hard" mean:

| Post (YOLOv3 / RetinaNet) | YOLOv11 (our models) | Consequence for the study |
|---|---|---|
| 3 anchor boxes per location; positives by IoU threshold | **Anchor-free**: 1 prediction per grid location, 8,400 per 640-px image; positives chosen by the **TaskAlignedAssigner** (top 10 locations per object) | Positives ≈ 10 per object, so the imbalance is ~1 : 1,000 rather than the post's counts |
| Separate **objectness** score + class score | **No objectness**: 9 class scores per location; "background" means all 9 are low | Every (location, class) pair is one sample: 8,400 × 9 = 75,600 per image |
| Hard 0/1 targets | **Soft targets** in [0, 1] for positives (alignment score ≈ box quality) | "Correctly classified" needs a threshold; loss never reaches 0 (the floor explained earlier) |
| γ = 2 (post's example: α = 0.25, γ = 2) | Our focal runs: **γ = 1.5, α = 0.25** | We report both settings in § 4.0 |

Two points in the post we will **not** reuse as stated: its YOLO snippet labels 5,259 RoIs as *positive* and 42 as *negative*, which contradicts its own text (background should dominate; the labels appear swapped); and its CE numbers (0.045 at x = 0.9, 1.0 at x = 0.1) use log base 10. We use natural log, as PyTorch and Ultralytics do, and say so wherever we quote the post.

## 3. Definitions used throughout

For each image, every (location, class) pair is a **sample** with predicted probability p and target t.

| Category | Rule | Meaning (post's terms) |
|---|---|---|
| **Positive** | t > 0 (location assigned to an object, that object's class) | foreground |
| **Negative** | t = 0 (all other pairs) | background, or wrong class |
| **Easy positive** | positive, p ≥ 0.5 | correctly classified foreground |
| **Hard positive** | positive, p < 0.5 | false negative: object scored as background |
| **Easy negative** | negative, p < 0.5 | true negative |
| **Hard negative** | negative, p ≥ 0.5 | false positive: background scored as an object |

0.5 is the post's "correctly classified" threshold. As a sensitivity check we repeat the split at **0.25** (Ultralytics' default detection confidence). Easy negatives are further split into p < 0.01 ("trivial") and 0.01–0.5, since the post's argument is about the many near-zero ones.

**Losses compared on the same predictions:** BCE(p, t) and focal FL(p, t) = α_t · (1 − p_t)^γ · BCE(p, t), with the exact implementation used in training (`water_debris_seg/yolo_focal.py`), plus the **gradient** each sample sends back, |∂L/∂logit|. Gradient share matters more than loss share, because it is what actually drives learning.

## 4. Analyses

### 4.0 Toy numbers and curves (no data)

- Table: BCE vs focal for an easy positive (p = 0.9), hard positive (p = 0.1), easy negative (p = 0.1), hard negative (p = 0.9), and a trivial negative (p = 0.001). Columns: BCE, focal (γ = 2, as in the post), focal (γ = 1.5, our runs), and the BCE/focal ratios. Reproduces the post's 360 vs 40 argument in natural log.
- Figure `focal_study/loss_curves.png`: loss vs p for y = 1 and y = 0, BCE and both focal settings, easy/hard regions shaded.

### 4.1 How imbalanced is LaRS for YOLOv11? (C1, C4)

Run the trained full-BCE model and Ultralytics' own assigner on LaRS val. Per image and in total:

- positive vs negative samples, and the ratio;
- positives per class (foreground–foreground imbalance; Float vs Boat/ship);
- figure `focal_study/fg_bg_counts.png`: bar chart (log scale) of negatives, positives, and positives per class.

### 4.2 Where does the loss go? (C2, C3)

For four checkpoints (full BCE, full focal, LoRA BCE, LoRA focal; stage-B `last.pt`), on LaRS val, using each model's predictions and its assigned targets:

| Table | Content |
|---|---|
| A | Per category (easy/hard × positive/negative, trivial split): number of samples, % of samples |
| B | Per category: **% of total BCE loss** and **% of total focal loss** on the *same* predictions (isolates the loss function's effect) |
| C | Per category: **% of total gradient** under BCE vs focal |
| D | Per category: mean loss per sample, BCE vs focal, and the ratio (the post's "CE/FL" on real data) |
| E | Each model with its own training loss: did focal training end up with fewer hard negatives / hard positives than BCE training? |

Figure `focal_study/loss_share.png`: stacked bars of loss and gradient share by category, BCE vs focal.

Expected (to be confirmed or refuted): under BCE, easy negatives hold a large share of the loss even though each is tiny; focal shifts the share toward hard samples. The open question is whether focal over-corrects: with α = 0.25 it also scales **positives down**, which could explain the drop on small and rare classes.

### 4.3 What does it do to the detections?

Full BCE vs full focal (LoRA added in the tables only), LaRS val, confidence 0.25, a detection counts as a true positive at IoU ≥ 0.5 with the right class:

- Counts per class: TP, FP, FN for each model.
- Figure `focal_study/score_hist.png`: confidence histograms of TP vs FP detections for each model (focal is known to produce lower, differently calibrated scores).
- **Detection images** `focal_study/examples/*.png`: side-by-side GT | BCE | focal; green = TP, red = FP, orange dashed = missed GT, labels with class and score.
- **Loss heatmaps** for the same images: per grid location, BCE loss vs focal loss (summed over classes), on the 80×80 stride-8 map, overlaid on the image. Shows where each loss "looks": spread over the water/sky background vs concentrated on objects.

**Image selection is fixed in advance** (no cherry-picking): every val image containing a Float (Float has 3 val boxes); the 3 images with the most small objects (< 32×32 px); the 3 images with the most objects inside a Static Obstacle box; and 3 random images (seed 0). About 12 images. If an image falls in several groups it is used once and the next one in that group is taken.

## 5. Data, models, protocol

| Item | Value |
|---|---|
| Images | LaRS val (198), the split already used for scoring. Analysis only: no training, tuning or checkpoint choice is made from it |
| Checkpoints | stage-B `last.pt` of `lars_full_stageB`, `lars_full_focal_stageB`, `lars_lora_stageB`, `lars_lora_focal_stageB` (SHA-256 recorded) |
| Input | 640 px letterboxed, no augmentation, same preprocessing as Ultralytics val |
| Targets | Ultralytics' TaskAlignedAssigner, run on each model's own predictions (as in training) |
| Focal settings | γ 1.5, α 0.25 (as trained); § 4.0 also shows γ 2 |
| Compute | Inference only; one short GPU job (~10 min) or CPU |

## 6. Outputs and audit trail

| Output | Path |
|---|---|
| Script | `water-debris-seg/scripts/focal_vs_bce_study.py` (reuses `FocalBCE` from `water_debris_seg/yolo_focal.py`) |
| Numbers | `focal_study/study.json`: every table above, checkpoint and GT hashes, image list with selection reason |
| Figures | `focal_study/*.png`, `focal_study/examples/*.png` |
| Report | `FOCAL_STUDY.md`: qualitative write-up for the presentation; every number and figure links back to `study.json` or the PNG it came from |
| Log | `focal_study/run_<job>.log` |

## 7. Limitations known up front

- Single seed per model; differences between models are indicative.
- "Easy/hard" depends on the 0.5 threshold; the 0.25 check shows how sensitive the split is.
- Soft targets blur the post's clean positive/negative picture: a positive with t = 0.4 and p = 0.45 is "hard" by the rule but almost perfectly predicted. Table D reports this case separately (|p − t| < 0.1).
- Float has 3 val boxes: its images are shown, but no Float statistics are interpretable.
