#!/usr/bin/env python3
"""Re-run EoMT on a small image set and save the raw masks behind the scored boxes.

Uses the exact model builders and inference functions of the benchmarked eval scripts
(eval_eomt_cansurf_mask2box.py, eval_eomt_cansurf_semantic_mask2box.py) with the configs
recorded in box_eval_registry.json `model`. Nothing here feeds a metric.

Modes
  boxsets  images in qualitative/selection.json, every EoMT run of their dataset.
           Boxes are re-derived from the masks and compared with the scored prediction
           file → masks/<dataset>/<run_id>/consistency.json (audit: are these the masks
           behind the numbers?).
  lars     all LaRS val images with the ab02 semantic and ab03 panoptic checkpoints.

Output: eval_outputs/water_debris_seg/qualitative/masks/<dataset>/<run_or_model>/<stem>.npz
  panoptic: class_map (int16, -1 = void), seg_map (int32)    semantic: class_map (uint8, argmax)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_outputs" / "water_debris_seg"
LARS_VAL = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/val/images")


def resolve(p: str) -> Path:
    return Path(p) if p.startswith("/") else REPO / p


def load_model(model_cfg: dict, device):
    kind, ckpt = model_cfg["kind"], resolve(model_cfg["ckpt"])
    if kind == "panoptic_coco133":
        from eval_eomt_cansurf_mask2box import build_model_coco133
        return build_model_coco133(ckpt, device)
    if kind == "panoptic_lars":
        from eval_eomt_cansurf_mask2box import build_model_lars_panoptic
        p = model_cfg["params"]
        return build_model_lars_panoptic(ckpt, device, num_classes=p["num_classes"], stuff_classes=p["stuff_classes"])
    if kind == "semantic_lars":
        from eval_eomt_cansurf_semantic_mask2box import build_model_lars_semantic
        return build_model_lars_semantic(ckpt, device)
    raise ValueError(kind)


def infer(model, kind: str, img_path: Path, device, params: dict):
    """Return (npz arrays, detections) using the benchmarked functions."""
    import torch
    from PIL import Image
    from torchvision import tv_tensors

    img = tv_tensors.Image(Image.open(img_path).convert("RGB"), dtype=torch.uint8)
    with torch.no_grad():
        if kind.startswith("panoptic"):
            from eval_eomt_cansurf_mask2box import predict_image, preds_to_detections
            preds = predict_image(model, img, device)
            dets = preds_to_detections(preds, stuff_classes=set(model.stuff_classes),
                                       num_classes=model.num_classes, category_id=0)
            arrays = {"class_map": preds[:, :, 0].numpy().astype(np.int16),
                      "seg_map": preds[:, :, 1].numpy().astype(np.int32)}
        else:
            from eval_eomt_cansurf_semantic_mask2box import obstacle_masks_to_detections, predict_semantic_logits
            logits = predict_semantic_logits(model, img, device)
            dets = obstacle_masks_to_detections(logits, category_id=0, min_pixels=params.get("min_pixels", 32))
            arrays = {"class_map": torch.argmax(logits, dim=0).numpy().astype(np.uint8)}
    return arrays, dets


def box_key(b) -> tuple:
    return tuple(round(float(v)) for v in b)


def run_boxsets(device) -> None:
    import torch

    registry = json.loads((EVAL / "box_eval_registry.json").read_text())
    sel = json.loads((EVAL / "qualitative" / "selection.json").read_text())
    picks_by_ds = defaultdict(list)
    for p in sel["picks"]:
        picks_by_ds[p["dataset"]].append(p)
    for run_id in sel["meta"]["runs"]:
        run = next(r for r in registry["runs"] if r["id"] == run_id)
        if "model" not in run:
            continue
        ds = run["dataset"]
        img_dir = Path(registry["ground_truth"][ds]["images"])
        saved = defaultdict(list)
        for d in json.loads((EVAL / run["pred"]).read_text()):
            saved[d["image_id"]].append(box_key(d["bbox"]))
        out = EVAL / "qualitative" / "masks" / ds / run_id
        out.mkdir(parents=True, exist_ok=True)
        model = load_model(run["model"], device)
        report = []
        for p in picks_by_ds[ds]:
            arrays, dets = infer(model, run["model"]["kind"], img_dir / p["file_name"], device, run["model"]["params"])
            np.savez_compressed(out / f"{Path(p['file_name']).stem}.npz", **arrays)
            ours = sorted(box_key(d["bbox"]) for d in dets)
            theirs = sorted(saved.get(p["image_id"], []))
            report.append({"image_id": p["image_id"], "file_name": p["file_name"], "n_rederived": len(ours),
                           "n_scored": len(theirs), "identical": ours == theirs})
        n_ok = sum(r["identical"] for r in report)
        (out / "consistency.json").write_text(json.dumps({
            "run_id": run_id, "pred": run["pred"], "model": run["model"], "n_images": len(report),
            "n_identical_box_sets": n_ok, "images": report}, indent=2))
        print(f"{run_id}: {n_ok}/{len(report)} images re-derive identical boxes")
        del model
        torch.cuda.empty_cache()


def run_lars(device) -> None:
    import torch

    registry = json.loads((EVAL / "box_eval_registry.json").read_text())
    cfgs = {}
    for r in registry["runs"]:
        if "model" in r and r["model"]["kind"] in ("panoptic_lars", "semantic_lars"):
            cfgs.setdefault(r["model"]["kind"], r["model"])
    images = sorted(p for p in LARS_VAL.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    for kind, cfg in cfgs.items():
        out = EVAL / "qualitative" / "masks" / "lars_val" / kind
        out.mkdir(parents=True, exist_ok=True)
        model = load_model(cfg, device)
        for img_path in images:
            arrays, _ = infer(model, kind, img_path, device, cfg["params"])
            np.savez_compressed(out / f"{img_path.stem}.npz", **arrays)
        (out / "provenance.json").write_text(json.dumps({"model": cfg, "images": str(LARS_VAL), "n": len(images)}, indent=2))
        print(f"lars_val {kind}: {len(images)} images")
        del model
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["boxsets", "lars", "all"])
    args = parser.parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode in ("boxsets", "all"):
        run_boxsets(device)
    if args.mode in ("lars", "all"):
        run_lars(device)


if __name__ == "__main__":
    main()
