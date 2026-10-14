#!/usr/bin/env python3
"""Write the 2026-10-13 corrections to Weights & Biases without altering any logged value.

WRITES to W&B (needs --write; without it only prints the plan). Two kinds of change:

1. New runs in project `water-debris-seg`, group `audit_2026-10-13`, job_type `audit`. Each run's
   summary is read from the audited local files (RESCORE.json, pq_*_28011322.json, logs, ...),
   and its config records what it supersedes/complements, the prediction/GT sha256, and this
   repo's git commit (the code is tracked now; the original runs never pinned it).
   Run ids are deterministic (`rescore_<registry_id>` ...), so re-running skips existing runs.
2. On the original runs: add a tag and append a note pointing at the corrected run. Summaries,
   configs and history are NOT modified; tags/notes can be removed in the W&B UI.

Run from anywhere inside the repo after sourcing ~/.config/water-debris-seg/wandb.env.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_outputs" / "water_debris_seg"
PROJECT = "water-debris-seg"
GROUP = "audit_2026-10-13"
YOLO_PROJECT = "-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo"
TAG_SCORER = "superseded-scorer-2026-10-13"
TAG_BRIDGE = "superseded-label-bridge-2026-10-13"
TAG_VALONLY = "val-split-only"
DECISION = "docs/water_debris_seg_decision_log.md (entries dated 2026-10-13)"
# registry runs whose GT matches what the original W&B run scored (CANSURF originals used the 2221-image GT)
ORIGINAL_GT_RUNS = {"flow_eomt_zeroshot", "flow_eomt_panoptic", "flow_eomt_semantic", "cansurf_eomt_zeroshot_origgt",
                    "cansurf_eomt_panoptic_origgt", "cansurf_eomt_semantic", "cansurf_eomt_semantic_min256"}


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def plan(entity: str, commit: str) -> tuple[list[dict], list[dict]]:
    """Return (new runs, edits to existing runs)."""
    rescore = json.loads((EVAL / "RESCORE.json").read_text())
    registry = {r["id"]: r for r in json.loads((EVAL / "box_eval_registry.json").read_text())["runs"]}
    base = {"audit": True, "repo_commit": commit, "decision_log": DECISION}
    new, edits = [], []

    def run_url(project: str, rid: str) -> str:
        return f"https://wandb.ai/{entity}/{project}/runs/{rid}"

    # A. box-metric re-scoring, one run per registry run
    for r in rescore["runs"]:
        reg = registry[r["id"]]
        old_proj = reg.get("wandb_project", PROJECT)
        is_eomt = r["method"].startswith("EoMT")
        summ = flat(r["new"], "")
        summ.update({f"legacy/{k}": v for k, v in r["legacy"].items() if v is not None})
        new.append({
            "id": f"rescore_{r['id']}", "tags": ["rescore", "pycocotools"],
            "summary": summ,
            "config": {**base, "kind": "box_rescore", "registry_id": r["id"], "dataset": r["dataset"], "method": r["method"],
                       "scorer": "pycocotools", "pred_file": r["pred"], "pred_sha256": r["pred_sha256"],
                       "gt_file": r["gt"], "gt_sha256": r["gt_sha256"],
                       "relation": "supersedes (scorer)" if is_eomt else "complements (Ultralytics value unaffected)",
                       "original_run": reg.get("wandb_run"), "original_run_url": run_url(old_proj, reg["wandb_run"]) if reg.get("wandb_run") else None},
            "notes": f"pycocotools re-score of {r['pred']}. " + ("The original run logged the legacy scorer's value (legacy/*)." if is_eomt else "")})
        # tag each original run once, pointing at the re-score on the GT version that run actually used
        if r["id"] in ORIGINAL_GT_RUNS and reg.get("wandb_run"):
            edits.append({"project": old_proj, "id": reg["wandb_run"], "tag": TAG_SCORER, "corrected_by": f"rescore_{r['id']}"})

    # B. ablation 01 corrected PQ and the reproduction check
    for rid, src, rel in [("ab01_zeroshot_lars_pq_predremap", "pq_zeroshot_predremap_28011322.json", "txjoa9ef"),
                          ("ab03_pq_script_check", "pq_native_ab03_check_28011322.json", "67kk33hk / qmi79nsh / xcl7zvm8")]:
        d = json.loads((EVAL / "ablations/01_zeroshot_lars/sources" / src).read_text())
        summ = {f"metrics/val_{k}": d[k] for k in ("pq_all", "sq_all", "rq_all", "pq_things", "pq_stuff")}
        summ.update({f"pq_class/{c}": v["pq"] for c, v in d["per_class"].items()})
        new.append({"id": rid, "tags": ["pq", "lars-native-space"], "summary": summ,
                    "config": {**base, "kind": "lars_pq", "slurm": "28011322", "mode": d["mode"], "ckpt": d["ckpt"],
                               "mapping": d.get("mapping"), "n_images": d["n_images"], "label_space": d["label_space"],
                               "relation": "supersedes" if "ab01" in rid else "reproduces", "original_run": rel,
                               "original_run_url": run_url(PROJECT, "txjoa9ef") if "ab01" in rid else None,
                               "source_file": f"ablations/01_zeroshot_lars/sources/{src}"},
                    "notes": ("Zero-shot COCO EoMT scored in native LaRS 11-class space by remapping predictions "
                              "(configs/coco133_to_lars_pred_map.json). Replaces the GT-bridge value 0.0014." if "ab01" in rid else
                              "Same script on the fine-tuned ab03 checkpoint; reproduces EoMT's own validate PQ (0.3945) — validation of the script.")})
    edits.append({"project": PROJECT, "id": "txjoa9ef", "tag": TAG_BRIDGE, "corrected_by": "ab01_zeroshot_lars_pq_predremap"})

    # C. YOLO FloW test-split result (Ultralytics) — the online training run only has the val split
    log = (EVAL / "ablations/08_specialized_yolo/sources/flow_test_val_27942882.log").read_text(errors="replace")
    m = re.search(r"\ball\s+800\s+2024\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", log)
    p, rcl, m50, m5095 = (float(x) for x in m.groups())
    new.append({"id": "ab08_yolo_flow_test_split", "tags": ["yolo", "test-split"],
                "summary": {"ultralytics/test/precision": p, "ultralytics/test/recall": rcl, "ultralytics/test/mAP50": m50,
                            "ultralytics/test/mAP50-95": m5095, "pycocotools/test/mAP": rescore_value(rescore, "flow_yolo11n", "mAP"),
                            "pycocotools/test/AP_small": rescore_value(rescore, "flow_yolo11n", "AP_small")},
                "config": {**base, "kind": "yolo_test_split", "slurm": "27942882", "split": "test (official 800 images, 2024 boxes)",
                           "relation": "complements", "original_run": "flow_train_20261012_153258",
                           "original_run_url": run_url(YOLO_PROJECT, "flow_train_20261012_153258"),
                           "source_file": "ablations/08_specialized_yolo/sources/flow_test_val_27942882.log",
                           "pycocotools_run": "rescore_flow_yolo11n"},
                "notes": "Test-split result of the FloW YOLOv11n weights. The original training run's summary is the 120-image val split."})
    edits.append({"project": YOLO_PROJECT, "id": "flow_train_20261012_153258", "tag": TAG_VALONLY, "corrected_by": "ab08_yolo_flow_test_split"})

    # D. VidEoMT per-keyframe AP (the eval runs have empty summaries)
    for job, wid in [("27772987", "lars_vitb_27772987"), ("27775358", "eval_27775358"), ("27806956", "eval_27806956")]:
        d = json.loads((EVAL / f"ablations/07_videomt_temporal/sources/videomt_rescore_keyframe_{job}.json").read_text())
        summ = flat({"per_keyframe": d["per_keyframe"]})
        summ.update(flat({"upstream_all_frames": d.get("upstream_all_frames", {})}))
        summ["n_preds"] = d["n_preds"]
        new.append({"id": f"ab07_keyframe_ap_{job}", "tags": ["videomt", "keyframe-ap"], "summary": summ,
                    "config": {**base, "kind": "videomt_keyframe_ap", "slurm": job, "relation": "complements", "original_run": wid,
                               "original_run_url": run_url(PROJECT, wid),
                               "source_file": f"ablations/07_videomt_temporal/sources/videomt_rescore_keyframe_{job}.json",
                               "metric": "per-keyframe VIS AP (split_ytvis_by_keyframe); not a temporal-ID metric"},
                    "notes": "Per-keyframe AP recomputed from the saved predictions; the original run logged no AP."})

    # E. qualitative error analysis summary
    rows = list(csv.DictReader((EVAL / "qualitative/error_summary.csv").open()))
    cons = [json.loads(p.read_text()) for p in (EVAL / "qualitative/masks").glob("*/*/consistency.json")]
    summ = {f"recall_iou50/{r['run_id']}": float(r["recall_iou50"]) for r in rows}
    for r in rows:
        for k, v in r.items():
            if k.startswith(("det_", "gt_")):
                summ[f"{k}/{r['run_id']}"] = int(v)
    summ["masks_reproduce_scored_boxes"] = sum(c["n_identical_box_sets"] for c in cons)
    summ["masks_checked"] = sum(c["n_images"] for c in cons)
    new.append({"id": "qualitative_error_analysis", "tags": ["qualitative"], "summary": summ,
                "config": {**base, "kind": "qualitative", "slurm": "28001374 (masks)", "relation": "new",
                           "source_file": "qualitative/error_summary.csv", "selection": "qualitative/selection.json"},
                "notes": "Per-object error counts (found/merged/missed/...) per run; see eval_outputs/water_debris_seg/qualitative/README.md."})
    return new, edits


def rescore_value(rescore: dict, rid: str, key: str) -> float:
    return next(r for r in rescore["runs"] if r["id"] == rid)["new"][key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="actually write to W&B (default: print the plan)")
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=60)
    entity = api.viewer.entity
    commit = git("rev-parse", "HEAD")
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "water-debris-seg/scripts/wandb_log_corrections.py"],
                             capture_output=True).returncode == 0
    new, edits = plan(entity, commit)
    print(f"entity={entity} project={PROJECT} group={GROUP} commit={commit[:10]} script_tracked={tracked}")
    print(f"{len(new)} new runs, {len(edits)} existing runs to tag")
    for n in new:
        print(f"  NEW  {n['id']:42} {len(n['summary'])} summary keys")
    for e in edits:
        print(f"  TAG  {e['project'][:22]:22}/{e['id']:30} +{e['tag']}  -> {e['corrected_by']}")
    if not args.write:
        print("dry run: nothing written (use --write)")
        return
    if not tracked:
        raise SystemExit("commit this script first so the audit runs pin code that contains it")

    for n in new:
        try:
            api.run(f"{entity}/{PROJECT}/{n['id']}")
            print(f"exists, skipped: {n['id']}")
            continue
        except Exception:
            pass
        run = wandb.init(project=PROJECT, entity=entity, group=GROUP, name=n["id"], id=n["id"], job_type="audit",
                         config=n["config"], tags=n["tags"], notes=n["notes"], reinit=True)
        for k, v in n["summary"].items():
            run.summary[k] = v
        if n["id"] == "qualitative_error_analysis":
            rows = list(csv.DictReader((EVAL / "qualitative/error_summary.csv").open()))
            run.log({"error_summary": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
        run.finish()
        print(f"created: {n['id']}")

    for e in edits:
        r = api.run(f"{entity}/{e['project']}/{e['id']}")
        url = f"https://wandb.ai/{entity}/{PROJECT}/runs/{e['corrected_by']}"
        changed = False
        if e["tag"] not in r.tags:
            r.tags = [*r.tags, e["tag"]]
            changed = True
        line = f"SUPERSEDED/QUALIFIED 2026-10-13 — corrected value: {url}"
        if line not in (r.notes or ""):
            r.notes = ((r.notes or "").rstrip() + "\n" + line).strip()
            changed = True
        if changed:
            r.update()
            print(f"tagged: {e['id']} (+{e['tag']})")
        else:
            print(f"already tagged: {e['id']}")


if __name__ == "__main__":
    main()
