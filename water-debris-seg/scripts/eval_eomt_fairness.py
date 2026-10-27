#!/usr/bin/env python3
"""EoMT scoring for fairness fixes 1 and 1b (plan: eval_outputs/water_debris_seg/fairness_fixes/FIX1_LARS_SPLIT_METRICS.md).

One forward pass per image, two EoMT post-processings:
  * panoptic (EoMT's ``to_per_pixel_preds_panoptic``, as in EoMT validation) -> LaRS PQ, and LaRS mIoU after merging the
    panoptic map to obstacle / water / sky (obstacle = Static Obstacle + the 8 object classes; verified to equal the
    LaRS semantic-mask "obstacle" on all 198 val images);
  * instance (EoMT's own ``mask_classification_instance.eval_step``: top-100 (query, class) pairs, score = class
    probability x mean mask probability) -> masks -> min/max boxes with real confidences -> pycocotools box AP.
For comparison, LaRS also gets box AP from the panoptic segments with the old constant score 0.5 (ablations 04/05/09).

--dataset lars     PQ (11 classes), mIoU (3), box AP on lars_val_gt_coco_boxes.json (9 classes, 8 objects, per class)
--dataset flow / cansurf
                   box AP, debris-only (main) and class-agnostic (reference); every prediction keeps its source class
--model coco133 | lars_panoptic | lars_semantic (semantic: LaRS mIoU only, used to validate the mIoU scorer)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "water-debris-seg"))

LARS_VAL = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/val")
LARS_GT = REPO / "eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/sources/lars_val_gt_coco_boxes.json"
BOX_SETS = {
    "flow": (REPO / "eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json",
             REPO / "data/water_debris/flow/test/images"),
    "cansurf": (REPO / "eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json",
                REPO / "data/water_debris/cansurf_home/val/images"),
}
TIGHT_MAP = REPO / "water-debris-seg/configs/coco133_to_lars_pred_map_tightened.json"
FULL_MAP = REPO / "water-debris-seg/configs/coco133_to_lars_pred_map.json"
LARS_NAMES = ["Static Obstacle", "Water", "Sky", "Boat/ship", "Row boats", "Paddle board", "Buoy", "Swimmer", "Animal",
              "Float", "Other"]
# LaRS 11-class index -> LaRS box-GT category id (YOLO order); Water / Sky have no boxes
LARS_TO_BOX = {0: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
D_LARS = {9, 10}                                     # Float, Other
D_COCO = {"bottle", "cup", "frisbee", "sports ball", "kite", "backpack", "handbag", "suitcase", "umbrella"}
STATS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium",
         "AR_large"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def forward(model, img, device):
    """Final-layer mask logits at image resolution (Q, H, W) and class logits (Q, C+1), as EoMT's eval_step."""
    import torch.nn.functional as F

    img = img.to(device)
    sizes = [img.shape[-2:]]
    mask_layers, class_layers = model(model.resize_and_pad_imgs_instance_panoptic([img]))
    m = F.interpolate(mask_layers[-1], model.img_size, mode="bilinear")
    m = model.revert_resize_and_pad_logits_instance_panoptic(m, sizes)[0]
    return m, class_layers[-1][0]


def instance_dets(mask_logits, class_logits, top_k=100):
    """EoMT eval_step instance post-processing -> [(class_index, score, (x, y, w, h))]."""
    import torch

    scores = class_logits.softmax(-1)[:, :-1]
    n_cls = scores.shape[-1]
    k = min(top_k, scores.numel())
    top_s, top_i = scores.flatten().topk(k, sorted=False)
    labels = top_i % n_cls
    q = top_i // n_cls
    ml = mask_logits[q]
    masks = ml > 0
    mscore = (ml.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
    final = top_s * mscore
    rows, cols = masks.any(2), masks.any(1)
    out = []
    for j in range(k):
        r = torch.nonzero(rows[j]).flatten()
        c = torch.nonzero(cols[j]).flatten()
        if r.numel() == 0:
            continue
        x0, x1, y0, y1 = int(c[0]), int(c[-1]), int(r[0]), int(r[-1])
        out.append((int(labels[j]), float(final[j]), (x0, y0, x1 - x0 + 1, y1 - y0 + 1)))
    return out


def panoptic_boxes(preds):
    """Panoptic segments -> [(class_index, (x, y, w, h))] (for the constant-score comparison)."""
    cls, seg = preds[:, :, 0].numpy(), preds[:, :, 1].numpy()
    out = []
    for sid in np.unique(seg):
        if sid < 0:
            continue
        m = seg == sid
        ys, xs = np.nonzero(m)
        out.append((int(cls[m][0]), (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))))
    return out


def cocoeval(gt_path, dets, cat_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dets:
        return {k: 0.0 for k in STATS}, None
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(str(gt_path))
        ev = COCOeval(g, g.loadRes(dets), "bbox")
        if cat_ids is not None:
            ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(zip(STATS, map(float, ev.stats))), ev


def per_class_ap(ev, names):
    p = ev.eval["precision"]
    out = {}
    for k, cid in enumerate(ev.params.catIds):
        pk = p[:, :, k, 0, -1]
        v = pk[pk > -1]
        out[names[cid]] = float(v.mean()) if v.size else -1.0
    return out


def load_coco_maps(path):
    """COCO contiguous index -> LaRS index (via the mapping json)."""
    from datasets.coco_panoptic import CLASS_MAPPING  # type: ignore[import-untyped]

    cfg = json.loads(Path(path).read_text())
    to_lars = {CLASS_MAPPING[e["coco_id"]]: cfg["lars_index"][lars] for lars, es in cfg["map"].items() for e in es}
    return to_lars, None


def coco_names_all():
    """COCO contiguous index -> name for the 80 object classes (Ultralytics' COCO list, same order as EoMT's 0..79)."""
    import yaml
    import ultralytics

    d = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml").read_text())
    return {int(k): v for k, v in d["names"].items()}


def _build_panoptic_untrained(device):
    """Same EoMT-B LaRS panoptic model as build_model_lars_panoptic, without loading a checkpoint."""
    from models.eomt import EoMT  # type: ignore[import-untyped]
    from models.vit import ViT  # type: ignore[import-untyped]
    from training.mask_classification_panoptic import MaskClassificationPanoptic  # type: ignore[import-untyped]

    network = EoMT(encoder=ViT(backbone_name="vit_base_patch14_reg4_dinov2", img_size=(640, 640), ckpt_path=None),
                   num_classes=11, num_q=200, num_blocks=3)
    return MaskClassificationPanoptic(network=network, img_size=(640, 640), num_classes=11, stuff_classes=[0, 1, 2],
                                      attn_mask_annealing_enabled=True, attn_mask_annealing_start_steps=[14782, 36955, 59128],
                                      attn_mask_annealing_end_steps=[29564, 51737, 73910], ckpt_path=None,
                                      load_ckpt_class_head=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["coco133", "lars_panoptic", "lars_semantic", "box_instance"], required=True,
                    help="box_instance: EoMT trained on FloW / CANSURF SAM masks (fix 3/4), single class, full or LoRA")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", choices=["lars", "flow", "cansurf"], required=True)
    ap.add_argument("--mapping", choices=["tightened", "full"], default="tightened", help="COCO -> LaRS map (coco133 on LaRS)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import tv_tensors

    from eval_eomt_cansurf_mask2box import build_model_coco133, build_model_lars_panoptic
    from eval_eomt_cansurf_semantic_mask2box import build_model_lars_semantic, predict_semantic_logits
    from eval_eomt_lars_panoptic_pq import NUM_LARS, STUFF, MetricHost, load_target, remap_coco_preds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"model": args.model, "ckpt": str(args.ckpt), "ckpt_sha256": sha(args.ckpt), "dataset": args.dataset,
           "eomt_post_processing": {"instance": "mask_classification_instance.eval_step, top-100", "panoptic": "to_per_pixel_preds_panoptic"}}
    if args.model == "coco133":
        model = build_model_coco133(args.ckpt, device)
        map_path = TIGHT_MAP if args.mapping == "tightened" else FULL_MAP
        to_lars, _ = load_coco_maps(map_path)
        coco_names = coco_names_all()
        res["mapping"] = {"file": str(map_path), "sha256": sha(map_path)}
    elif args.model == "lars_panoptic":
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        lora_keys = [k for k in sd if k.endswith("lora_A")]
        if lora_keys:      # LoRA checkpoint (fix 4): rebuild the LoRA layers before loading
            from water_debris_seg.eomt_lora import inject_lora

            r = int(sd[lora_keys[0]].shape[0])
            scale = float(sd[lora_keys[0].replace("lora_A", "lora_scale")])
            model = _build_panoptic_untrained(device)
            inject_lora(model.network, r, scale * r)
            model.load_state_dict(sd, strict=True)
            model = model.eval().to(device)
            res["lora"] = {"r": r, "alpha": scale * r, "lora_tensors": len(lora_keys)}
        else:
            model = build_model_lars_panoptic(args.ckpt, device, num_classes=NUM_LARS, stuff_classes=STUFF)
    elif args.model == "box_instance":
        from models.eomt import EoMT  # type: ignore[import-untyped]
        from models.vit import ViT  # type: ignore[import-untyped]
        from training.mask_classification_instance import MaskClassificationInstance  # type: ignore[import-untyped]

        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        net = EoMT(encoder=ViT(backbone_name="vit_base_patch14_reg4_dinov2", img_size=(640, 640), ckpt_path=None),
                   num_classes=1, num_q=200, num_blocks=3)
        model = MaskClassificationInstance(network=net, img_size=(640, 640), num_classes=1, attn_mask_annealing_enabled=False,
                                           ckpt_path=None, load_ckpt_class_head=False)
        lora_keys = [k for k in sd if k.endswith("lora_A")]
        if lora_keys:
            from water_debris_seg.eomt_lora import inject_lora

            r = int(sd[lora_keys[0]].shape[0])
            scale = float(sd[lora_keys[0].replace("lora_A", "lora_scale")])
            inject_lora(model.network, r, scale * r)
            res["lora"] = {"r": r, "alpha": scale * r, "lora_tensors": len(lora_keys)}
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
    else:
        model = build_model_lars_semantic(args.ckpt, device)
    model.eval()

    def lars_index(c):          # predicted class -> LaRS 11-class index (None = ignored)
        return c if args.model == "lars_panoptic" else to_lars.get(c)

    def load(path):
        return tv_tensors.Image(Image.open(path).convert("RGB"), dtype=torch.uint8)

    if args.dataset == "lars":
        pan = json.loads((LARS_VAL / "panoptic_annotations.json").read_text())
        anns = sorted(pan["annotations"], key=lambda a: a["file_name"])[: args.limit or None]
        gt_boxes = json.loads(LARS_GT.read_text())
        img_id = {Path(im["file_name"]).stem: im["id"] for im in gt_boxes["images"]}
        conf = np.zeros((4, 3), dtype=np.int64)          # rows: pred obstacle/water/sky/void; cols: GT obstacle/water/sky
        host = MetricHost(); host.metrics.to(device)
        dets, dets_const = [], []
        from training.lightning_module import LightningModule  # type: ignore[import-untyped]

        for k, a in enumerate(anns):
            stem = Path(a["file_name"]).stem
            img = load(LARS_VAL / "images" / f"{stem}.jpg")
            sem_gt = np.asarray(Image.open(LARS_VAL / "semantic_masks" / f"{stem}.png"))
            with torch.no_grad():
                if args.model == "lars_semantic":
                    pred3 = torch.argmax(predict_semantic_logits(model, img, device), 0).numpy()
                else:
                    ml, cl = forward(model, img, device)
                    preds = model.to_per_pixel_preds_panoptic([ml], cl[None], model.stuff_classes, model.mask_thresh,
                                                              model.overlap_thresh)[0].cpu()
                    if args.model == "coco133":
                        preds = remap_coco_preds(preds, to_lars)
                    target, crowd = load_target(stem, a)
                    LightningModule.update_metrics_panoptic(host, [preds.to(device)], [target.to(device)], [crowd.to(device)], 0)
                    c11 = preds[:, :, 0].numpy()
                    pred3 = np.full(c11.shape, 3)
                    pred3[(c11 == 0) | ((c11 >= 3) & (c11 < NUM_LARS))] = 0
                    pred3[c11 == 1] = 1
                    pred3[c11 == 2] = 2
                    for c, s, box in instance_dets(ml, cl):
                        li = lars_index(c)
                        if li in LARS_TO_BOX:
                            dets.append({"image_id": img_id[stem], "category_id": LARS_TO_BOX[li], "bbox": list(box),
                                         "score": s, "src_class": c})
                    for li, box in panoptic_boxes(preds):
                        if li in LARS_TO_BOX:
                            dets_const.append({"image_id": img_id[stem], "category_id": LARS_TO_BOX[li], "bbox": list(box), "score": 0.5})
            v = sem_gt != 255
            np.add.at(conf, (pred3[v], sem_gt[v]), 1)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(anns)}", flush=True)
        iou = [conf[c, c] / (conf[c, :].sum() + conf[:, c].sum() - conf[c, c]) for c in range(3)]
        res["mIoU"] = {"mean": float(np.mean(iou)), "per_class": dict(zip(["obstacle", "water", "sky"], map(float, iou))),
                       "void_pixel_share": float(conf[3].sum() / conf.sum()), "gt": "LaRS val semantic_masks (0 obstacle, 1 water, 2 sky, 255 ignored)"}
        res["n_images"] = len(anns)
        if args.model != "lars_semantic":
            r = host.metrics[0].compute()[:-1]
            pq = r[:, 0]
            order = sorted(host.metrics[0].cat_id_to_continuous_id, key=host.metrics[0].cat_id_to_continuous_id.get)[:-1]
            res["PQ"] = {"all": float(pq.mean()), "things": float(pq[[i for i, c in enumerate(order) if c >= 3]].mean()),
                         "stuff": float(pq[[i for i, c in enumerate(order) if c < 3]].mean()),
                         "per_class": {LARS_NAMES[c]: float(pq[i]) for i, c in enumerate(order)}}
            names = {c["id"]: c["name"] for c in gt_boxes["categories"]}
            objs = [c for c in names if names[c] != "Static Obstacle"]
            all9, ev = cocoeval(LARS_GT, dets)
            res["box_AP"] = {"all_9_classes": all9, "objects_8_classes": cocoeval(LARS_GT, dets, objs)[0],
                             "per_class_AP": per_class_ap(ev, names) if ev else {},
                             "constant_score_0.5": {"all_9_classes": cocoeval(LARS_GT, dets_const)[0],
                                                    "objects_8_classes": cocoeval(LARS_GT, dets_const, objs)[0]},
                             "n_dets": len(dets), "gt": str(LARS_GT), "gt_sha256": sha(LARS_GT)}
            pred_file = args.out.with_suffix(".preds.json")
            pred_file.write_text(json.dumps(dets))
            res["box_AP"]["predictions"] = str(pred_file)
    else:
        gt_path, img_dir = BOX_SETS[args.dataset]
        gt = json.loads(Path(gt_path).read_text())
        ims = gt["images"][: args.limit or None]
        debris, agnostic = [], []
        names = coco_names_all() if args.model == "coco133" else ({0: args.dataset} if args.model == "box_instance" else dict(enumerate(LARS_NAMES)))
        for k, im in enumerate(ims):
            with torch.no_grad():
                ml, cl = forward(model, load(img_dir / im["file_name"]), device)
                for c, s, box in instance_dets(ml, cl):
                    is_obj = c < 80 if args.model == "coco133" else (c == 0 if args.model == "box_instance" else 3 <= c < NUM_LARS)
                    if not is_obj:
                        continue
                    d = {"image_id": im["id"], "category_id": 0, "bbox": list(box), "score": s, "src_class": names.get(c, str(c))}
                    agnostic.append(d)
                    if (args.model == "coco133" and names.get(c) in D_COCO) or (args.model == "lars_panoptic" and c in D_LARS) \
                            or args.model == "box_instance":           # in-domain single class: debris-only = class-agnostic
                        debris.append(d)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(ims)}", flush=True)
        res["n_images"] = len(ims)
        res["debris_classes"] = sorted(D_COCO) if args.model == "coco133" else ([args.dataset] if args.model == "box_instance" else ["Float", "Other"])
        res["box_AP"] = {"debris_only": cocoeval(gt_path, debris)[0], "class_agnostic": cocoeval(gt_path, agnostic)[0],
                         "n_dets_debris": len(debris), "n_dets_agnostic": len(agnostic), "gt": str(gt_path), "gt_sha256": sha(gt_path)}
        pred_file = args.out.with_suffix(".preds.json")
        pred_file.write_text(json.dumps(agnostic))
        res["box_AP"]["predictions_class_agnostic_with_src_class"] = str(pred_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k in ("PQ", "mIoU", "box_AP")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
