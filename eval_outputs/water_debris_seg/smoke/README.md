# Smoke eval tier

Quick sanity before zero-shot / fine-tune (8–16 images, overlays).

| Stack | Output dir | Status |
|---|---|---|
| EoMT | `eomt/validate_log.txt` | **done** job **27594939** (2 val batches; PQ/metrics ≈ 0 — COCO head + 8 LaRS frames, no maritime ckpt) |
| VidEoMT | `videomt/` | not run |
| YOLOv11 | `yolo/microrun/` | **done** job **27591918** (1-epoch micro-train; full-val mAP50≈0.10) |
