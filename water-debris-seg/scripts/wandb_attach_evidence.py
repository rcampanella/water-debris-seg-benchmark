#!/usr/bin/env python3
"""Give the audit_2026-10-13 correction runs real tracked data on W&B.

The correction runs written by wandb_log_corrections.py only set summary values: no logged step
(empty charts) and no evidence files. This script resumes ONLY those runs (group audit_2026-10-13,
project water-debris-seg) and, without changing any value:

  1. logs the run's numeric summary as one history step, so the values appear in charts;
  2. attaches the files the values were computed from as an artifact `evidence-<run_id>`
     (prediction files, GT, metrics JSONs, PQ / keyframe JSONs, logs, mapping configs).

Original runs are never opened. Idempotent: a run that already has its evidence artifact is skipped.
WRITES to W&B only with --write (default prints the plan).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_outputs" / "water_debris_seg"
WDS = REPO / "water-debris-seg"
PROJECT, GROUP = "water-debris-seg", "audit_2026-10-13"


def evidence_files(run_id: str, rescore: dict, registry: dict) -> list[Path]:
    """Files whose content produced the run's values."""
    if run_id.startswith("rescore_"):
        rid = run_id.removeprefix("rescore_")
        row = rescore[rid]
        metrics = EVAL / registry[rid]["metrics"]
        files = [EVAL / row["pred"], EVAL / row["gt"], metrics]
        legacy = metrics.with_name(metrics.name.removesuffix(".json") + ".legacy.json")
        return files + ([legacy] if legacy.is_file() else [])
    src = EVAL / "ablations"
    table = {
        "ab01_zeroshot_lars_pq_predremap": [src / "01_zeroshot_lars/sources/pq_zeroshot_predremap_28011322.json",
                                            WDS / "configs/coco133_to_lars_pred_map.json", WDS / "configs/panoptic_coco_categories.json",
                                            src / "01_zeroshot_lars/sources/validate_27600047.log"],
        "ab03_pq_script_check": [src / "01_zeroshot_lars/sources/pq_native_ab03_check_28011322.json",
                                 src / "03_finetune_lars_panoptic/sources/validate_27660922.log"],
        "ab08_yolo_flow_test_split": [src / "08_specialized_yolo/sources/flow_test_val_27942882.log",
                                      src / "08_specialized_yolo/sources/yolo11n_flow_test_preds_coco.json",
                                      src / "08_specialized_yolo/sources/yolo11n_flow_test_preds_coco.metrics.json",
                                      src / "08_specialized_yolo/sources/flow_data.yaml", src / "08_specialized_yolo/sources/flow_val_split.txt"],
        "qualitative_error_analysis": [EVAL / "qualitative/error_summary.csv", EVAL / "qualitative/selection.json",
                                       EVAL / "qualitative/README.md"],
    }
    if run_id.startswith("ab07_keyframe_ap_"):
        job = run_id.removeprefix("ab07_keyframe_ap_")
        return [src / f"07_videomt_temporal/sources/videomt_rescore_keyframe_{job}.json"]
    return table[run_id]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=90)
    entity = api.viewer.entity
    rescore = {r["id"]: r for r in json.loads((EVAL / "RESCORE.json").read_text())["runs"]}
    registry = {r["id"]: r for r in json.loads((EVAL / "box_eval_registry.json").read_text())["runs"]}
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    runs = list(api.runs(f"{entity}/{PROJECT}", filters={"group": GROUP}))
    plan = []
    for r in runs:
        files = evidence_files(r.id, rescore, registry)
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            raise SystemExit(f"{r.id}: missing evidence files {missing}")
        has_art = any(a.name.split(":")[0] == f"evidence-{r.id}" for a in r.logged_artifacts())
        numeric = {k: v for k, v in r.summary.items() if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)}
        plan.append((r, files, has_art, numeric))
        print(f"{'SKIP' if has_art else 'PLAN'} {r.id:40} log {len(numeric):3} values @ step {r.lastHistoryStep + 1}, "
              f"artifact {len(files)} files ({sum(f.stat().st_size for f in files) / 1e6:.1f} MB)")
    if not args.write:
        print("dry run: nothing written (use --write)")
        return

    for r, files, has_art, numeric in plan:
        if has_art:
            continue
        run = wandb.init(project=PROJECT, entity=entity, id=r.id, resume="must")
        run.log(numeric)
        art = wandb.Artifact(f"evidence-{r.id}", type="evidence",
                             description="Files the run's values were computed from (audit_2026-10-13).",
                             metadata={"repo_commit_at_upload": head,
                                       "files": {str(f.relative_to(REPO)): sha(f) for f in files}})
        for f in files:
            art.add_file(str(f), name=str(f.relative_to(REPO)))
        run.log_artifact(art, aliases=["audit-2026-10-13"])
        run.config.update({"evidence_artifact": f"evidence-{r.id}"}, allow_val_change=True)
        run.finish()
        print(f"done: {r.id}")


if __name__ == "__main__":
    main()
