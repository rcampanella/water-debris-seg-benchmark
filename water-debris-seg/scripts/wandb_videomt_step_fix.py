#!/usr/bin/env python3
"""Correction runs for the two VidEoMT training runs whose W&B charts show no curves (ablation 07).

Upstream train_net_video.py calls define_metric("*", step_metric="trainer/global_step"), a Lightning key that
Detectron2 never writes, so every logged point sits at trainer/global_step = 0 and no chart draws a line. The values
themselves are intact (synced from TensorBoard, with the real iteration in `global_step`).

WRITES to W&B (needs --write; without it only prints the plan and runs the checks):
1. New run per original, id `stepfix_<original id>`, group `audit_2026-10-29`, job_type `audit`: the original's
   history (losses, lr, segm/*) re-logged against `global_step` (define_metric), merged into one row per iteration,
   plus an evidence artifact with the local training log. Every loss value is first checked against that log
   (Detectron2 prints them rounded to 4 significant digits); the run aborts on a mismatch.
2. On the original run: a tag and a note pointing at the correction run. Its history, summary and config are not
   modified.
Idempotent: an existing `stepfix_*` run is skipped.

segm/* semantics (recorded in each run's config): the 4k run (27772987) evaluated with upstream all-frames VIS AP,
which is 0 by construction on LaRS keyframe-only labels, so every segm value is 0; its per-keyframe AP (0.016) is in
run ab07_keyframe_ap_27775358. The 16k run (27776810) used the per-keyframe eval patch; segm/* is in percent
(Detectron2), e.g. segm/AP 0.98 = 0.0098.

Run after sourcing ~/.config/water-debris-seg/wandb.env.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "eval_outputs/water_debris_seg/ablations/07_videomt_temporal/sources"
PROJECT = "water-debris-seg"
GROUP = "audit_2026-10-29"
TAG = "step-axis-fixed-2026-10-29"
RUNS = {
    "lars_vitb_27772987": {"log": SRC / "videomt_train_27772987.log", "iters": 4000,
                           "segm": "upstream all-frames VIS AP: 0 by construction on LaRS (single-frame GT tracks); "
                                   "per-keyframe AP 0.016 is in run ab07_keyframe_ap_27775358"},
    "lars_vitb_16k_27776810": {"log": SRC / "videomt_train_16k_27776810.log", "iters": 16000,
                               "segm": "per-keyframe VIS AP (keyframe eval patch), in percent; final segm/AP 0.98 = "
                                       "0.0098; eval-only rescore with up to 50 tracks: run ab07_keyframe_ap_27806956"},
}
SKIP = {"_runtime", "_step", "_timestamp", "trainer/global_step", "global_step"}
LOSS_RE = re.compile(r"iter: (\d+)\s+(total_loss: .*?)\s+time:")


def log_losses(path: Path) -> dict[int, dict[str, float]]:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = LOSS_RE.search(line)
        if m:
            out[int(m.group(1))] = {k: float(v) for k, v in re.findall(r"(\w+): ([-\d.e]+)", m.group(2))}
    return out


def build(api, entity: str, rid: str, spec: dict) -> tuple[list[dict], dict]:
    run = api.run(f"{entity}/{PROJECT}/{rid}")
    rows = defaultdict(dict)
    for r in run.scan_history(page_size=5000):
        vals = {k: v for k, v in r.items() if k not in SKIP and v is not None and not k.startswith("_")}
        if vals:
            rows[int(r["global_step"])].update(vals)
    hist = [{"global_step": s, **v} for s, v in sorted(rows.items())]
    # every loss value against the local training log (W&B global_step = Detectron2 iter; log prints 4 sig. digits)
    ref, checked = log_losses(spec["log"]), 0
    for h in hist:
        if "total_loss" not in h:
            continue
        line = ref[h["global_step"]]
        for k, v in line.items():
            if k.startswith("loss") or k == "total_loss":
                if abs(h[k] - v) > max(5e-4 * abs(v), 1e-4):
                    raise SystemExit(f"{rid}: {k} at step {h['global_step']}: W&B {h[k]} vs log {v}")
                checked += 1
    n_loss = sum("total_loss" in h for h in hist)
    n_segm = sum("segm/AP" in h for h in hist)
    assert n_loss == len(ref), (rid, n_loss, len(ref))
    assert hist[-1]["global_step"] == spec["iters"], (rid, hist[-1]["global_step"])
    return hist, {"n_rows": len(hist), "n_loss_rows": n_loss, "n_segm_rows": n_segm, "values_checked_vs_log": checked,
                  "original_url": run.url, "original_state": run.state}


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    api = wandb.Api(timeout=120)
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for rid, spec in RUNS.items():
        hist, info = build(api, entity, rid, spec)
        new_id = f"stepfix_{rid}"
        print(rid, info)
        if not args.write:
            continue
        try:
            api.run(f"{entity}/{PROJECT}/{new_id}")
            print(f"  {new_id} exists, skipped")
            continue
        except wandb.errors.CommError:
            pass
        cfg = {"audit": True, "kind": "step_axis_fix", "original_run": rid, "original_run_url": info["original_url"],
               "reason": "upstream define_metric('*', step_metric='trainer/global_step'); that key is never written "
                         "by Detectron2, so all points sit at 0. Values re-logged unchanged against global_step.",
               "segm_meaning": spec["segm"], "evidence_log": str(spec["log"].relative_to(REPO)),
               "repo_commit": commit, **{k: v for k, v in info.items() if k != "original_url"},
               "decision_log": "docs/water_debris_seg_decision_log.md (2026-10-29)"}
        new = wandb.init(project=PROJECT, entity=entity, id=new_id, name=new_id, group=GROUP, job_type="audit",
                         tags=["audit", "videomt", "step-axis-fix"], config=cfg,
                         notes=f"Re-log of {rid} against global_step (original charts plot at trainer/global_step = 0). "
                               f"segm: {spec['segm']}.")
        new.define_metric("global_step")
        new.define_metric("*", step_metric="global_step")
        for h in hist:
            new.log(h)
        art = wandb.Artifact(f"evidence-{new_id}", type="evidence", description=f"local training log of {rid}")
        art.add_file(str(spec["log"]))
        new.log_artifact(art)
        new.finish()
        orig = api.run(f"{entity}/{PROJECT}/{rid}")
        if TAG not in orig.tags:
            orig.tags = [*orig.tags, TAG]
        note = f"Charts: see {new_id} (same values against global_step; this run plots at trainer/global_step = 0)."
        if note not in (orig.notes or ""):
            orig.notes = ((orig.notes or "") + "\n" + note).strip()
        orig.update()
        print(f"  wrote {new_id}; tagged {rid}")


if __name__ == "__main__":
    main()
