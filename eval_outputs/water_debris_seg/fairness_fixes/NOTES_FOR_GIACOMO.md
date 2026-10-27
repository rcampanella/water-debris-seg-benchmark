# Points to raise with Giacomo (2026-10-23)

Decisions taken to make the model comparison fair and coherent ([README.md](README.md)). Each needs Giacomo's view before the results are reported.

| # | Decision | Why | Trade-off to discuss |
|---|---|---|---|
| 1 | **YOLO + SAM instead of YOLO-seg** for YOLO's LaRS PQ and mIoU | A box detector cannot be scored with PQ / mIoU (boxes cannot be turned into masks). YOLO + SAM needs no extra model type: YOLOv11s is retrained with Water and Sky as box classes, a frozen SAM turns each box into a mask | The masks come from SAM, a model much larger than EoMT. Mitigation: an **EoMT panoptic + SAM control** row (same mask source) separates detection quality from mask quality. YOLO-seg was the alternative: masks from YOLO itself, coarser (1/4 resolution) |
| 2 | **Only YOLOv11s** in the comparison | Matches the CANSURF paper (`yolo11s.pt`, 150 epochs, batch 64, [`train.py`](https://github.com/ZaidAljundiHW2/CANSURF)); on LaRS it beats YOLOv11n (object mAP 0.176 vs 0.145) | YOLOv11n would show model-scale effects and map to edge devices. Its LaRS results remain in ablation 10; the existing FloW / CANSURF YOLO results are YOLOv11n (ablation 08), not the paper's recipe as previously stated |
| 3 | **YOLOv11s, not YOLO26** | CANSURF comparability; all YOLO code and results are on v11 (which uses Distribution Focal Loss for boxes) | YOLO26 drops DFL, is NMS-free and newer; a YOLO26s reference row could be added later (~1–1.5 h per dataset) |
| 4 | **EoMT semantic dropped from the main table** | One EoMT model (panoptic) gives PQ, mIoU and box AP, like YOLOv11s + SAM; the benchmark becomes coherent | Semantic's mIoU (98.0) is LaRS's classic obstacle / water / sky task. Condition: panoptic-derived mIoU within ~1 point; ablation 02 stays as a reference |
| 5 | **VidEoMT kept in a separate temporal table** | It needs LaRS video and outputs per-keyframe mask AP and flicker, not PQ / mIoU / box AP | Its flicker (8 object classes) is not yet like-for-like with EoMT panoptic's (all classes); an aligned re-computation is inference only |
| 6 | **LoRA vs full fine-tuning for EoMT only** (fix 4); LoRA discarded for YOLO | EoMT is ~10× larger than YOLOv11s; on small datasets LoRA may match full fine-tuning with fewer trainable weights. YOLO LoRA (ablation 10) trailed full fine-tuning and is not pursued | Ablation 03 never reached EoMT's mask-free inference (annealing schedule longer than the run); both new arms scale it to the run length |

Also worth mentioning:

- **EoMT box AP so far used a constant confidence (0.5)** in ablations 04, 05, 09, understating EoMT against YOLO. Fix 1b re-scores with EoMT's own instance scores.
- **On FloW / CANSURF every predicted object counted as debris** (for zero-shot: all 80 COCO classes, including cars and giraffes). New rule: debris-only classes as the main number, class-agnostic as reference.
- **Fix 3 reverses the original "hard rule"** (EoMT trained only on LaRS): EoMT is trained on FloW / CANSURF using SAM masks prompted by the ground-truth boxes, after a SAM mask quality check.
- **YOLO zero-shot has never been tested** while EoMT zero-shot has, on all three datasets.
