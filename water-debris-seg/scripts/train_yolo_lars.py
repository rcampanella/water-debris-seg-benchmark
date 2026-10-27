#!/usr/bin/env python3
"""Train YOLOv11n on LaRS boxes traced from the panoptic masks (ablation 10).

Two variants x two stages:
  --variant full   all weights trained (standard Ultralytics fine-tuning from yolo11n.pt)
  --variant lora   LoRA on backbone+neck convs, head + BatchNorm trained (water_debris_seg/yolo_lora.py)
  --stage A        train on train90, monitor the held-out sequences (held10) every epoch; pick the epoch count
  --stage B        retrain on the full train set (held10 included) for the stage-A epoch count; keep last.pt
  --loss focal     focal classification loss instead of BCE (water_debris_seg/yolo_focal.py); runs get a
                   `_focal` suffix and stage B reads the epoch count from the matching focal stage-A run

LaRS val is never used during training or selection; score it afterwards with eval_yolo_lars.py.
Runs are reachable at <ablation>/runs/<name> (a link to Ultralytics' save dir) and log to W&B project
`water-debris-seg` (Ultralytics names the W&B project after its `project=` argument).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "water-debris-seg"))
ABL = REPO / "eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes"
DATA = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/yolo_boxes")
FIX = REPO / "eval_outputs/water_debris_seg/fairness_fixes"
# --dataset: data root with stageA/B.yaml, output folder (runs/, sources/), run-name prefix, W&B group
DATASETS = {
    "lars": (DATA, ABL, "lars", "ablation10_yolo_lars"),
    "flow": (REPO / "data/water_debris/yolo_splits/flow", FIX, "flow", "fix3_sam_box_datasets"),
    "cansurf": (REPO / "data/water_debris/yolo_splits/cansurf", FIX, "cansurf", "fix3_sam_box_datasets"),
}


def select_epochs(results_csv: Path) -> dict:
    """Epoch with the best held-out fitness (Ultralytics: 0.1 mAP50 + 0.9 mAP50-95) in a stage-A run."""
    rows = list(csv.DictReader(results_csv.open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    fit = [0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows]
    b = max(range(len(rows)), key=lambda i: fit[i])
    pick = lambda r: {k: float(r[k]) for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss")}
    return {"source": str(results_csv), "epochs_run": len(rows), "best_epoch": b + 1, "best_fitness": fit[b],
            "at_best": pick(rows[b]), "at_last": pick(rows[-1])}


LOSS_TERMS = ("box_loss", "cls_loss", "dfl_loss")


def add_total_loss(trainer) -> None:
    """on_fit_epoch_end, run before the W&B callback: add train/total_loss and val/total_loss to the logged metrics.

    Each logged term already carries its gain (box 7.5, cls 0.5, dfl 1.5), so the total is their plain sum, i.e. the
    per-image loss the optimizer minimises. Skipped for the post-training best.pt validation, which has no val losses.
    """
    m = trainer.metrics
    if not all(f"val/{t}" in m for t in LOSS_TERMS):
        return
    tr = trainer.label_loss_items(trainer.tloss, prefix="train")
    m["train/total_loss"] = round(sum(float(tr[f"train/{t}"]) for t in LOSS_TERMS), 5)
    m["val/total_loss"] = round(sum(float(m[f"val/{t}"]) for t in LOSS_TERMS), 5)
    m["epoch"] = trainer.epoch + 1   # shared x-axis with runs backfilled by wandb_backfill_total_loss.py (W&B step = epoch)


def make_trainer(lora: bool, focal: tuple[float, float] | None):
    """DetectionTrainer that injects LoRA and/or switches to focal loss right after the pretrained weights load."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from water_debris_seg.yolo_focal import use_focal
    from water_debris_seg.yolo_lora import inject_lora

    class Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if focal is not None:
                use_focal(model, *focal)
                print(f"[focal] gamma={focal[0]} alpha={focal[1]}", flush=True)
            if lora:
                r, alpha = int(os.environ.get("LORA_R", 8)), float(os.environ.get("LORA_ALPHA", 16))
                print(f"[LoRA] r={r} alpha={alpha} {inject_lora(model, r, alpha)}", flush=True)
            return model

    return Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="lars")
    ap.add_argument("--data-root", type=Path, default=None, help="override the dataset's stageA/B.yaml folder (e.g. LaRS 11-class)")
    ap.add_argument("--variant", choices=["full", "lora"], required=True)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--epochs", type=int, default=100, help="stage A epoch budget")
    ap.add_argument("--epochs-from", type=Path, help="stage A results.csv (stage B)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction (smoke tests only)")
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal-gamma", type=float, default=1.5)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained weights, e.g. yolo11s.pt")
    ap.add_argument("--lr0", type=float, default=None,
                    help="explicit AdamW lr; replicates optimizer=auto (AdamW, momentum 0.9, warmup_bias_lr 0) with only the lr changed")
    ap.add_argument("--tag", default=None, help="run tag replacing <variant>[_focal] in run names and epoch_selection_<tag>.json")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                    help="image cache; 'ram' avoids re-reading images from slow scratch every epoch (2026-10-23)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO, settings
    settings.update({"wandb": True})
    tag = args.tag or (f"{args.variant}_focal" if args.loss == "focal" else args.variant)
    data_root, out_dir, prefix, group = DATASETS[args.dataset]
    data_root = args.data_root or data_root
    name = args.name or f"{prefix}_{tag}_stage{args.stage}"
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    os.chdir(runs)
    os.environ.setdefault("WANDB_RUN_GROUP", group)

    epochs, selection = args.epochs, None
    if args.stage == "B":
        selection = select_epochs(args.epochs_from)
        epochs = selection["best_epoch"]
        (out_dir / "sources").mkdir(parents=True, exist_ok=True)
        (out_dir / "sources" / f"epoch_selection_{prefix}_{tag}.json" if args.dataset != "lars"
         else out_dir / "sources" / f"epoch_selection_{tag}.json").write_text(json.dumps(selection, indent=2))
        print(f"[stage B] {tag}: {epochs} epochs from {args.epochs_from}", flush=True)

    kw = dict(data=str(data_root / f"stage{args.stage}.yaml"), epochs=epochs, imgsz=args.imgsz, batch=args.batch,
              seed=args.seed, deterministic=True, device=0, project="water-debris-seg", name=name, exist_ok=True,
              fraction=args.fraction, plots=True, cache=False if args.cache == "none" else args.cache)
    if args.lr0 is not None:   # same as optimizer=auto picks for these run lengths (AdamW, 0.9, no bias warm-up), lr changed
        kw.update(optimizer="AdamW", lr0=args.lr0, momentum=0.9, warmup_bias_lr=0.0)
    focal = (args.focal_gamma, args.focal_alpha) if args.loss == "focal" else None
    if args.variant == "lora" or focal:
        kw["trainer"] = make_trainer(args.variant == "lora", focal)
    model = YOLO(args.model)
    model.callbacks["on_fit_epoch_end"].insert(0, add_total_loss)   # before W&B logs trainer.metrics
    model.train(**kw)
    actual = Path(model.trainer.save_dir).resolve()   # Ultralytics nests relative projects under runs/<task>/
    out = runs / name                                 # stable path used by the job scripts
    if out.is_symlink() or out.exists():
        out.unlink()
    out.symlink_to(actual, target_is_directory=True)
    (out / "run_spec.json").write_text(json.dumps({"variant": args.variant, "stage": args.stage, "dataset": args.dataset, "tag": tag, "cache": args.cache, "model": args.model,
                                                   "lr0": args.lr0 if args.lr0 is not None else "auto (AdamW 0.000769)", "loss": args.loss,
                                                   "focal_gamma": focal[0] if focal else None,
                                                   "focal_alpha": focal[1] if focal else None, "epochs": epochs,
                                                   "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed,
                                                   "lora_r": os.environ.get("LORA_R", "8") if args.variant == "lora" else None,
                                                   "lora_alpha": os.environ.get("LORA_ALPHA", "16") if args.variant == "lora" else None,
                                                   "data": kw["data"], "selection": selection}, indent=2))
    print("run dir:", out)


if __name__ == "__main__":
    main()
