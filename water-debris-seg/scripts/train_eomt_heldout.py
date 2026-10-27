#!/usr/bin/env python3
"""EoMT training with the held-out protocol, full fine-tuning or LoRA (fairness fix 4; fix 3 adds FloW / CANSURF).

  --stage A   train on LaRS train minus held10, validate on held10 every epoch (budget --epochs, default 12 as ablation 03);
              the epoch with the best held10 PQ sets the epoch count
  --stage B   retrain on all LaRS train for that epoch count; held10 monitored only; last checkpoint kept
  --arm full | lora (r=16, alpha=32 on every encoder block's qkv / proj / fc1 / fc2; water_debris_seg/eomt_lora.py)

Same starting weights as ablation 03 (COCO-panoptic EoMT-B 640 2x, class head re-initialised) and EoMT's default
optimizer. Attention-mask annealing is scaled to this run's length (same fractions as EoMT's COCO recipe).
Logs to W&B (group fix4_eomt_lora) and to a local CSV used for the stage-A selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "water-debris-seg"))
from water_debris_seg.eomt_lora import annealing_steps  # noqa: E402

FIX = REPO / "eval_outputs/water_debris_seg/fairness_fixes"
EOMT = REPO / "data/water_debris/third_party/eomt"
CKPT = REPO / "data/water_debris/checkpoints/eomt_coco/coco_panoptic_eomt_base_640_2x.bin"
DATASETS = {
    # zips (stage A / B), train images per stage, data class, num_classes, stuff classes
    "lars": (REPO / "data/water_debris/lars_eomt_zips", {"A": 2360, "B": 2605},
             "water_debris_seg.eomt_adapters.lars_panoptic.LarsPanoptic", 11, [0, 1, 2]),
    # fix 3: SAM pseudo-masks, single class, instance segmentation (stage sizes = YOLOv11s splits)
    "flow": (REPO / "data/water_debris/eomt_box_zips/flow", {"A": 1080, "B": 1200},
             "water_debris_seg.eomt_adapters.box_instance.BoxSetInstance", 1, None),
    "cansurf": (REPO / "data/water_debris/eomt_box_zips/cansurf", {"A": 4553, "B": 4950},
                "water_debris_seg.eomt_adapters.box_instance.BoxSetInstance", 1, None),
}
MODEL_CLS = {("panoptic", "full"): "training.mask_classification_panoptic.MaskClassificationPanoptic",
             ("panoptic", "lora"): "water_debris_seg.eomt_lora_modules.LoRAMaskClassificationPanoptic",
             ("instance", "full"): "training.mask_classification_instance.MaskClassificationInstance",
             ("instance", "lora"): "water_debris_seg.eomt_lora_modules.LoRAMaskClassificationInstance"}
METRICS = {"panoptic": "metrics/val_pq_all", "instance": "metrics/val_ap_all"}   # held10 selection metric
METRIC = METRICS["panoptic"]


def select_epochs(metrics_csv: Path, metric: str = METRIC) -> dict:
    """Epoch with the best held10 metric (PQ on LaRS, mask AP on the SAM pseudo-masks) in a stage-A run."""
    rows = [r for r in csv.DictReader(metrics_csv.open()) if r.get(metric)]
    vals = [(int(float(r["epoch"])) + 1, float(r[metric])) for r in rows]
    best = max(vals, key=lambda v: v[1])
    return {"source": str(metrics_csv), "metric": metric, "per_epoch": vals, "best_epoch": best[0], "best_value": best[1]}


BASE_LR = 1e-4  # EoMT MaskClassificationPanoptic / Instance default, used by every run before --lr-mult


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="lars")
    ap.add_argument("--arm", choices=["full", "lora"], required=True)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--epochs", type=int, default=12, help="stage A budget (ablation 03 used 12)")
    ap.add_argument("--epochs-from", type=Path, help="stage A metrics.csv (stage B)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lr-mult", type=float, default=1.0, help="multiplies EoMT's base lr (1e-4); LLRD still applies")
    ap.add_argument("--tag", default="", help="variant name added to the run dir (keeps the default runs intact)")
    ap.add_argument("--limit-batches", type=float, default=1.0, help="smoke tests only")
    args = ap.parse_args()

    zips, n_train, data_cls, n_cls, stuff = DATASETS[args.dataset]
    task = "panoptic" if stuff is not None else "instance"
    variant = f"{args.arm}_{args.tag}" if args.tag else args.arm
    name = f"eomt_{args.dataset}_{variant}_stage{args.stage}"
    run_dir = FIX / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs, selection = args.epochs, None
    if args.stage == "B":
        selection = select_epochs(args.epochs_from, METRICS[task])
        epochs = selection["best_epoch"]
        (FIX / "sources" / f"epoch_selection_eomt_{args.dataset}_{variant}.json").write_text(json.dumps(selection, indent=2))
    steps = (n_train[args.stage] // args.batch) * epochs
    start, end = annealing_steps(steps)
    # starting weights: COCO-panoptic EoMT-B, class head re-initialised (as ablation 03)
    init = {"ckpt_path": str(CKPT), "load_ckpt_class_head": False, "attn_mask_annealing_enabled": True, "lr": BASE_LR * args.lr_mult,
            "attn_mask_annealing_start_steps": start, "attn_mask_annealing_end_steps": end,
            "network": {"class_path": "models.eomt.EoMT", "init_args": {
                "num_q": 200, "num_blocks": 3,
                "encoder": {"class_path": "models.vit.ViT", "init_args": {"backbone_name": "vit_base_patch14_reg4_dinov2"}}}}}
    if args.arm == "lora":
        init.update(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    cfg = {"trainer": {"max_epochs": epochs, "devices": 1, "limit_train_batches": args.limit_batches,
                       "limit_val_batches": args.limit_batches, "default_root_dir": str(run_dir),
                       "logger": [{"class_path": "lightning.pytorch.loggers.WandbLogger", "init_args": {
                           "project": os.environ.get("WANDB_PROJECT", "water-debris-seg"), "group": "fix4_eomt_lora" if (args.dataset == "lars" or args.arm == "lora") else "fix3_sam_box_datasets",
                           "name": f"{name}_{os.environ.get('SLURM_JOB_ID', 'local')}", "save_dir": str(run_dir),
                           "log_model": False}},
                                  {"class_path": "lightning.pytorch.loggers.CSVLogger", "init_args": {"save_dir": str(run_dir), "name": "csv"}}]},
           "model": {"class_path": MODEL_CLS[(task, args.arm)], "init_args": init},
           "data": {"class_path": data_cls, "init_args": {"path": str(zips / f"stage{args.stage}"), "num_classes": n_cls,
                                                          "batch_size": args.batch, "num_workers": 4, "img_size": [640, 640],
                                                          **({"stuff_classes": stuff} if stuff is not None else {})}}}
    import yaml

    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "run_spec.json").write_text(json.dumps({"dataset": args.dataset, "task": task, "arm": args.arm, "stage": args.stage,
                                                       "epochs": epochs, "steps": steps, "lr": BASE_LR * args.lr_mult, "tag": args.tag, "annealing_start": start,
                                                       "annealing_end": end, "selection": selection, "eomt": str(EOMT),
                                                       "ckpt": str(CKPT)}, indent=2))
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'water-debris-seg'}:{EOMT}:{os.environ.get('PYTHONPATH', '')}")
    subprocess.run([sys.executable, "main.py", "fit", "--config", str(cfg_path)], cwd=EOMT, env=env, check=True)
    print("run dir:", run_dir)


if __name__ == "__main__":
    main()
