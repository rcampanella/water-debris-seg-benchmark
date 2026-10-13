#!/usr/bin/env python3
"""LaRS semantic EoMT (obstacle class) on CANSURF val → connected-components mask→box + mAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_coco_bbox import evaluate_bbox_map

try:
    from water_debris_seg.paths import scratch_root
except ModuleNotFoundError:

    def scratch_root() -> Path:
        import os

        return Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")


IMG_SIZE = (640, 640)
OBSTACLE_CLASS = 0


def build_model_lars_semantic(lightning_ckpt: Path, device, *, num_classes: int = 3):
    import torch
    from models.eomt import EoMT  # type: ignore[import-untyped]
    from models.vit import ViT  # type: ignore[import-untyped]
    from training.mask_classification_semantic import (  # type: ignore[import-untyped]
        MaskClassificationSemantic,
    )

    encoder = ViT(
        backbone_name="vit_base_patch14_reg4_dinov2",
        img_size=IMG_SIZE,
        ckpt_path=None,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=200,
        num_blocks=3,
    )
    model = MaskClassificationSemantic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=num_classes,
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[6520, 13040, 19560, 26080],
        attn_mask_annealing_end_steps=[13040, 19560, 26080, 32600],
        ckpt_path=None,
        load_ckpt_class_head=False,
        ignore_idx=255,
    )
    ckpt = torch.load(lightning_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model.to(device)


def predict_semantic_logits(model, img_tensor, device):
    import torch
    import torch.nn.functional as F

    img = img_tensor.to(device)
    imgs = [img]
    img_sizes = [img.shape[-2:]]
    crops, origins = model.window_imgs_semantic(imgs)
    mask_logits_per_layer, class_logits_per_layer = model(crops)
    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]
    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
    crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
    return model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0].cpu()


def obstacle_masks_to_detections(logits, *, category_id: int, min_pixels: int):
    import numpy as np
    import torch
    from masks_to_coco_boxes import mask_to_xywh
    from scipy.ndimage import label

    pred = torch.argmax(logits, dim=0).numpy()
    obstacle = pred == OBSTACLE_CLASS
    if not obstacle.any():
        return []
    labeled, n = label(obstacle)
    dets: list[dict] = []
    for seg_id in range(1, n + 1):
        comp = labeled == seg_id
        if comp.sum() < min_pixels:
            continue
        box = mask_to_xywh(comp)
        if box is None:
            continue
        dets.append({"category_id": category_id, "bbox": box, "score": 0.5})
    return dets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=scratch_root() / "data/cansurf/val/images",
    )
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--lightning-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pred-coco", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=32)
    parser.add_argument("--batch-log-every", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="ablation09_semantic_cansurf")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(
            project="water-debris-seg",
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "lightning_ckpt": str(args.lightning_ckpt),
                "obstacle_class": OBSTACLE_CLASS,
                "eval_only": args.pred_coco is not None,
            },
        )

    n_images = 0
    if args.pred_coco is not None:
        pred_file = args.pred_coco
        n_predictions = len(json.loads(pred_file.read_text(encoding="utf-8")))
    else:
        import torch
        from PIL import Image
        from torchvision import tv_tensors

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("Warning: running on CPU will be very slow", file=sys.stderr)

        model = build_model_lars_semantic(args.lightning_ckpt, device)
        with args.gt_coco.open(encoding="utf-8") as f:
            gt = json.load(f)
        id_by_name = {im["file_name"]: im["id"] for im in gt["images"]}

        image_paths = sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if args.limit > 0:
            image_paths = image_paths[: args.limit]
        n_images = len(image_paths)

        predictions: list[dict] = []
        ann_id = 1
        for i, img_path in enumerate(image_paths):
            if img_path.name not in id_by_name:
                continue
            image_id = id_by_name[img_path.name]
            pil = Image.open(img_path).convert("RGB")
            img = tv_tensors.Image(pil, dtype=torch.uint8)
            logits = predict_semantic_logits(model, img, device)
            for det in obstacle_masks_to_detections(
                logits,
                category_id=0,
                min_pixels=args.min_pixels,
            ):
                predictions.append({"id": ann_id, "image_id": image_id, **det})
                ann_id += 1
            if (i + 1) % args.batch_log_every == 0:
                print(f"processed {i + 1}/{len(image_paths)} images, {len(predictions)} dets")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(predictions), encoding="utf-8")
        pred_file = args.out
        n_predictions = len(predictions)

    metrics = evaluate_bbox_map(args.gt_coco, pred_file)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if args.pred_coco is None:
        print(f"wrote {args.out} ({n_predictions} boxes)")

    if args.wandb:
        import wandb

        wandb.log({**metrics, "n_predictions": n_predictions, "n_images": n_images})
        wandb.finish()


if __name__ == "__main__":
    main()
