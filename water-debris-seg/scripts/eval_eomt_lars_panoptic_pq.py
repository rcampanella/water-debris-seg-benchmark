#!/usr/bin/env python3
"""LaRS val panoptic PQ in native LaRS label space, for native or COCO-133 EoMT checkpoints.

Replaces the ablation 01 GT bridge (GT remapped to COCO ids; ids were wrong, see decision log
2026-10-13). Here the GT stays native LaRS and the *predictions* are remapped:

  --mode native   LaRS 11-class Lightning ckpt (ab03); no remap. Used to validate this script
                  against EoMT's own Lightning `validate` (ab03 val_pq_all ≈ 0.395).
  --mode coco133  frozen COCO-133 ckpt (ab01); predicted classes mapped many-to-one with
                  configs/coco133_to_lars_pred_map.json, stuff segments of one LaRS class merged.

Prediction = predict_image() from eval_eomt_cansurf_mask2box.py (last block, same
to_per_pixel_preds_panoptic as EoMT eval_step). Metric = torchmetrics PanopticQuality updated by
EoMT's own LightningModule.update_metrics_panoptic; pq_all = mean over the 11 LaRS classes
(EoMT drops the extra no-prediction class with [:-1]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LARS_VAL = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/val")
CAT_TO_IDX = {1: 0, 3: 1, 5: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 9, 19: 10}
LARS_NAMES = ["Static Obstacle", "Water", "Sky", "Boat/ship", "Row boats", "Paddle board", "Buoy",
              "Swimmer", "Animal", "Float", "Other"]
NUM_LARS = 11
STUFF = [0, 1, 2]
THINGS = list(range(3, NUM_LARS))


class MetricHost:
    """Minimal stand-in for EoMT's LightningModule state used by update_metrics_panoptic."""

    def __init__(self):
        import torch.nn as nn
        from torchmetrics.detection import PanopticQuality

        self.num_classes = NUM_LARS
        self.metrics = nn.ModuleList([PanopticQuality(THINGS, STUFF + [NUM_LARS],
                                                      return_sq_and_rq=True, return_per_class=True)])


def load_target(stem: str, ann: dict):
    import torch
    from PIL import Image

    rgb = np.asarray(Image.open(LARS_VAL / "panoptic_masks" / f"{stem}.png").convert("RGB")).astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]
    target = -torch.ones((*ids.shape, 2), dtype=torch.long)
    crowd = []
    for k, seg in enumerate(ann["segments_info"]):
        m = torch.from_numpy(ids == seg["id"])
        target[:, :, 0][m] = CAT_TO_IDX[seg["category_id"]]
        target[:, :, 1][m] = k
        crowd.append(bool(seg.get("iscrowd", 0)))
    return target, torch.tensor(crowd + [False])


def remap_coco_preds(preds, idx_map: dict[int, int]):
    """COCO-133 class channel → LaRS; unmapped/no-prediction → NUM_LARS; merge stuff per class."""
    import torch

    cls, seg = preds[:, :, 0].clone(), preds[:, :, 1].clone()
    lut = torch.full((134,), NUM_LARS, dtype=torch.long)
    for c, l in idx_map.items():
        lut[c] = l
    new_cls = lut[cls.clamp(0, 133)]
    new_seg = seg.clone()
    for s in STUFF:  # one segment per stuff class, as EoMT does for stuff
        new_seg[new_cls == s] = 100000 + s
    new_seg[new_cls == NUM_LARS] = -1
    return torch.stack([new_cls, new_seg], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["native", "coco133"], required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="Lightning .ckpt (native) or COCO .bin (coco133)")
    parser.add_argument("--mapping", type=Path, default=REPO / "water-debris-seg/configs/coco133_to_lars_pred_map.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from eval_eomt_cansurf_mask2box import build_model_coco133, build_model_lars_panoptic, predict_image
    from PIL import Image
    from torchvision import tv_tensors
    from training.lightning_module import LightningModule  # type: ignore[import-untyped]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "native":
        model = build_model_lars_panoptic(args.ckpt, device, num_classes=NUM_LARS, stuff_classes=STUFF)
        idx_map = None
    else:
        from datasets.coco_panoptic import CLASS_MAPPING  # type: ignore[import-untyped]

        model = build_model_coco133(args.ckpt, device)
        cfg = json.loads(args.mapping.read_text())
        idx_map = {CLASS_MAPPING[e["coco_id"]]: cfg["lars_index"][lars] for lars, es in cfg["map"].items() for e in es}

    pan = json.loads((LARS_VAL / "panoptic_annotations.json").read_text())
    anns = sorted(pan["annotations"], key=lambda a: a["file_name"])
    if args.limit:
        anns = anns[: args.limit]
    host = MetricHost()
    host.metrics.to(device)  # EoMT updates PQ on the GPU; torch.unique on CPU is ~100x slower
    for k, ann in enumerate(anns):
        stem = Path(ann["file_name"]).stem
        img = tv_tensors.Image(Image.open(LARS_VAL / "images" / f"{stem}.jpg").convert("RGB"), dtype=torch.uint8)
        with torch.no_grad():
            preds = predict_image(model, img, device)
        if idx_map is not None:
            preds = remap_coco_preds(preds, idx_map)
        target, crowd = load_target(stem, ann)
        LightningModule.update_metrics_panoptic(host, [preds.to(device)], [target.to(device)], [crowd.to(device)], 0)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(anns)}")

    result = host.metrics[0].compute()[:-1]
    pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]
    # torchmetrics orders per-class output by sorted category id: things 3..10 then stuff 0..2? use its map
    order = sorted(host.metrics[0].cat_id_to_continuous_id, key=host.metrics[0].cat_id_to_continuous_id.get)[:-1]
    per_class = {LARS_NAMES[c]: {"pq": float(pq[i]), "sq": float(sq[i]), "rq": float(rq[i])} for i, c in enumerate(order)}
    th = [i for i, c in enumerate(order) if c in THINGS]
    st = [i for i, c in enumerate(order) if c in STUFF]
    out = {
        "mode": args.mode, "ckpt": str(args.ckpt), "mapping": str(args.mapping) if idx_map else None,
        "n_images": len(anns), "label_space": "LaRS native 11-class (GT not remapped)",
        "pq_all": float(pq.mean()), "sq_all": float(sq.mean()), "rq_all": float(rq.mean()),
        "pq_things": float(pq[th].mean()), "pq_stuff": float(pq[st].mean()), "per_class": per_class,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
