#!/usr/bin/env python3
"""Per-frame EoMT preds on LaRS val clips → consecutive-frame IoU (flicker proxy)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inventory_lars_sequences import clip_id_from_name

try:
    from water_debris_seg.paths import scratch_root
except ModuleNotFoundError:

    def scratch_root() -> Path:
        import os

        return Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")


IMG_SIZE = (640, 640)


def build_model_semantic(lightning_ckpt: Path, device):
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
    network = EoMT(encoder=encoder, num_classes=3, num_q=200, num_blocks=3)
    model = MaskClassificationSemantic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=3,
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


def build_model_panoptic(lightning_ckpt: Path, device, *, num_classes: int = 11, stuff_classes: list[int] | None = None):
    import torch
    from models.eomt import EoMT  # type: ignore[import-untyped]
    from models.vit import ViT  # type: ignore[import-untyped]
    from training.mask_classification_panoptic import (  # type: ignore[import-untyped]
        MaskClassificationPanoptic,
    )

    stuff_classes = stuff_classes or [0, 1, 2]
    encoder = ViT(
        backbone_name="vit_base_patch14_reg4_dinov2",
        img_size=IMG_SIZE,
        ckpt_path=None,
    )
    network = EoMT(encoder=encoder, num_classes=num_classes, num_q=200, num_blocks=3)
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


def predict_panoptic_class_map(model, img_tensor, device):
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
    return preds_list[0][:, :, 0].cpu()


def predict_semantic_logits(model, img_tensor, device):
    import torch
    import torch.nn.functional as F

    img = img_tensor.to(device)
    crops, origins = model.window_imgs_semantic([img])
    mask_logits_per_layer, class_logits_per_layer = model(crops)
    mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
    crop_logits = model.to_per_pixel_logits_semantic(
        mask_logits, class_logits_per_layer[-1]
    )
    return model.revert_window_logits_semantic(crop_logits, origins, [img.shape[-2:]])[0]


def mean_iou(a, b, ignore: int = 255) -> float:
    import torch

    valid = (a != ignore) & (b != ignore)
    if not valid.any():
        return float("nan")
    a, b = a[valid], b[valid]
    ious = []
    for c in torch.unique(a):
        if c.item() == ignore:
            continue
        m1, m2 = a == c, b == c
        inter = (m1 & m2).sum().float()
        union = (m1 | m2).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-dir",
        type=Path,
        default=scratch_root() / "data/lars/val/images_seq",
    )
    parser.add_argument("--lightning-ckpt", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("semantic", "panoptic"),
        default="semantic",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all parsed clips")
    parser.add_argument("--max-frames-per-clip", type=int, default=0)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision import tv_tensors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "panoptic":
        model = build_model_panoptic(args.lightning_ckpt, device)
    else:
        model = build_model_semantic(args.lightning_ckpt, device)

    by_clip: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(args.seq_dir.glob("*.jpg")):
        cid = clip_id_from_name(p.name)
        if cid and cid != "_unparsed":
            by_clip[cid].append(p)

    clip_ids = sorted(by_clip.keys())
    if args.max_clips > 0:
        clip_ids = clip_ids[: args.max_clips]

    clip_results = []
    all_flicker = []

    for cid in clip_ids:
        frames = sorted(by_clip[cid], key=lambda x: x.name)
        if args.max_frames_per_clip > 0:
            frames = frames[: args.max_frames_per_clip]
        if len(frames) < 2:
            continue
        ious = []
        prev_pred = None
        for fp in frames:
            pil = Image.open(fp).convert("RGB")
            img = tv_tensors.Image(pil, dtype=torch.uint8)
            if args.task == "panoptic":
                pred = predict_panoptic_class_map(model, img, device)
            else:
                logits = predict_semantic_logits(model, img, device)
                pred = logits.argmax(dim=0)
            if prev_pred is not None:
                ious.append(mean_iou(prev_pred.cpu(), pred.cpu()))
            prev_pred = pred
        valid = [x for x in ious if x == x]
        flicker = 1.0 - (sum(valid) / len(valid)) if valid else float("nan")
        clip_results.append(
            {
                "clip_id": cid,
                "n_frames": len(frames),
                "mean_pairwise_iou": sum(valid) / len(valid) if valid else None,
                "flicker_proxy": flicker,
            }
        )
        if valid:
            all_flicker.append(flicker)

    summary = {
        "n_clips": len(clip_results),
        "mean_flicker_proxy": sum(all_flicker) / len(all_flicker) if all_flicker else None,
        "clips": clip_results,
        "ckpt": str(args.lightning_ckpt),
        "task": args.task,
        "note": f"{args.task} flicker = 1 - mean consecutive per-class mIoU",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_clips", "mean_flicker_proxy")}, indent=2))


if __name__ == "__main__":
    main()
