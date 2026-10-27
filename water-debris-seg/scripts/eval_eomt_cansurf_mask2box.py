#!/usr/bin/env python3
"""EoMT COCO panoptic zero-shot on CANSURF val → mask→box COCO detections + mAP."""

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


COCO133_STUFF = list(range(80, 133))
IMG_SIZE = (640, 640)


def build_model_coco133(ckpt_path: Path, device):
    import torch
    from models.eomt import EoMT  # type: ignore[import-untyped]
    from models.vit import ViT  # type: ignore[import-untyped]
    from training.mask_classification_panoptic import (  # type: ignore[import-untyped]
        MaskClassificationPanoptic,
    )

    encoder = ViT(
        backbone_name="vit_base_patch14_reg4_dinov2",
        img_size=IMG_SIZE,
        ckpt_path=None,
    )
    network = EoMT(
        encoder=encoder,
        num_classes=133,
        num_q=200,
        num_blocks=3,
    )
    model = MaskClassificationPanoptic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=133,
        stuff_classes=COCO133_STUFF,
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[29564, 73910, 118256],
        attn_mask_annealing_end_steps=[59128, 103474, 147820],
        ckpt_path=str(ckpt_path),
        load_ckpt_class_head=True,
    )
    model.eval()
    return model.to(device)


def build_model_lars_panoptic(
    lightning_ckpt: Path,
    device,
    *,
    num_classes: int,
    stuff_classes: list[int],
):
    import torch
    from models.eomt import EoMT  # type: ignore[import-untyped]
    from models.vit import ViT  # type: ignore[import-untyped]
    from training.mask_classification_panoptic import (  # type: ignore[import-untyped]
        MaskClassificationPanoptic,
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
    model = MaskClassificationPanoptic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=num_classes,
        stuff_classes=stuff_classes,
        attn_mask_annealing_enabled=True,
        attn_mask_annealing_start_steps=[14782, 36955, 59128],
        attn_mask_annealing_end_steps=[29564, 51737, 73910],
        ckpt_path=None,
        load_ckpt_class_head=False,
    )
    ckpt = torch.load(lightning_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model.to(device)


def predict_image(model, img_tensor, device):
    import torch
    import torch.nn.functional as F

    img = img_tensor.to(device)
    imgs = [img]
    img_sizes = [img.shape[-2:]]
    transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
    mask_logits_per_layer, class_logits_per_layer = model(transformed)
    mask_logits = mask_logits_per_layer[-1]
    class_logits = class_logits_per_layer[-1]
    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, img_sizes
    )[0]
    preds_list = model.to_per_pixel_preds_panoptic(
        [mask_logits],
        class_logits,
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )
    return preds_list[0].cpu()


def preds_to_detections(preds, *, stuff_classes: set[int], num_classes: int, category_id: int):
    import numpy as np
    from masks_to_coco_boxes import mask_to_xywh

    class_map = preds[:, :, 0].numpy()
    seg_map = preds[:, :, 1].numpy()
    dets: list[dict] = []
    for seg_id in np.unique(seg_map):
        if seg_id < 0:
            continue
        mask = seg_map == seg_id
        if not mask.any():
            continue
        cls = int(class_map[mask][0])
        if cls in stuff_classes or cls >= num_classes:
            continue
        box = mask_to_xywh(mask)
        if box is None:
            continue
        dets.append({"category_id": category_id, "bbox": box, "score": 0.5})
    return dets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/water_debris/cansurf_home/val/images",  # on home since 2026-10-23 (scratch purge)
    )
    parser.add_argument(
        "--gt-coco",
        type=Path,
        required=True,
        help="COCO detection JSON (e.g. cansurf_val_gt_coco.json)",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=scratch_root() / "artifacts/checkpoints/eomt/coco_panoptic_eomt_base_640_2x.bin",
        help="COCO-133 .bin checkpoint (default zero-shot path)",
    )
    parser.add_argument(
        "--lightning-ckpt",
        type=Path,
        default=None,
        help="LaRS panoptic fine-tune Lightning .ckpt (ablation 03)",
    )
    parser.add_argument("--num-classes", type=int, default=133)
    parser.add_argument(
        "--stuff-classes",
        default="80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132",
        help="Comma-separated class ids treated as stuff (excluded from boxes)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pred-coco",
        type=Path,
        default=None,
        help="Skip inference; evaluate existing prediction JSON",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-log-every", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="ablation05_eomt_cansurf_zeroshot")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(
            project="water-debris-seg",
            group=args.wandb_group,
            name=args.wandb_name,
            config={
                "ckpt": str(args.ckpt),
                "lightning_ckpt": str(args.lightning_ckpt) if args.lightning_ckpt else None,
                "num_classes": args.num_classes,
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

        if args.lightning_ckpt is not None:
            stuff_list = [int(x.strip()) for x in args.stuff_classes.split(",") if x.strip()]
            model = build_model_lars_panoptic(
                args.lightning_ckpt,
                device,
                num_classes=args.num_classes,
                stuff_classes=stuff_list,
            )
        else:
            model = build_model_coco133(args.ckpt, device)
        stuff = set(model.stuff_classes)

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
            preds = predict_image(model, img, device)
            for det in preds_to_detections(
                preds,
                stuff_classes=stuff,
                num_classes=model.num_classes,
                category_id=0,
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
