#!/usr/bin/env python3
"""Reconcile what Weights & Biases holds online with the numbers reported in the repo.

READ-ONLY: queries the W&B API with the key from WANDB_API_KEY (source
~/.config/water-debris-seg/wandb.env first); never writes to W&B.

Inputs : eval_outputs/water_debris_seg/{wandb_result_map.json, box_eval_registry.json, RESCORE.json}
Outputs: eval_outputs/water_debris_seg/wandb_runs.json   every benchmark run: url, state, job id, summary
         eval_outputs/water_debris_seg/WANDB_AUDIT.md    result -> run table with a verdict per value

Verdicts: matches current | online = superseded (legacy) value | differs | not logged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_outputs" / "water_debris_seg"
PROJECT_RE = re.compile(r"^(water-debris-seg|videomt|-home-rcampanella-projects-water-debris-seg-benchmark.*)$")
SKIP_PREFIX = ("_", "system", "gradients", "parameters")


def url(entity: str, project: str, run_id: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def clean(d, limit=None) -> dict:
    out = {}
    for k, v in d.items():
        if k.startswith(SKIP_PREFIX) or isinstance(v, bool) or not isinstance(v, (int, float, str)):
            continue
        out[k] = v
    return out


def value_of(spec: dict, rescore: dict) -> float | None:
    kind = spec["kind"]
    if kind == "rescore":
        row = next(r for r in rescore["runs"] if r["id"] == spec["registry_id"])
        return row[spec["which"]][spec["metric"]]
    if kind == "json":
        v = json.loads((EVAL / spec["file"]).read_text())
        for k in spec["path"]:
            v = v[k]
        return float(v)
    if kind == "log":
        m = re.search(spec["regex"], (EVAL / spec["file"]).read_text(errors="replace"))
        return float(m.group(1)) if m else None
    raise ValueError(kind)


def verdict(online, current, legacy, tol) -> str:
    if online is None:
        return "not logged"
    if current is not None and abs(online - current) <= tol:
        return "matches current"
    if legacy is not None and abs(online - legacy) <= tol:
        return "online = superseded (legacy) value"
    return "differs"


def fmt(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None, help="default: API viewer entity")
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=60)
    entity = args.entity or api.viewer.entity
    rmap = json.loads((EVAL / "wandb_result_map.json").read_text())
    registry = {r["id"]: r for r in json.loads((EVAL / "box_eval_registry.json").read_text())["runs"]}
    rescore = json.loads((EVAL / "RESCORE.json").read_text())

    # 1. every benchmark run
    runs = []
    for p in api.projects(entity):
        if not PROJECT_RE.match(p.name):
            continue
        for r in api.runs(f"{entity}/{p.name}"):
            m = re.search(r"(\d{7,8})$", r.name) or re.search(r"_(\d{7,8})", r.name)
            runs.append({
                "id": r.id, "name": r.name, "project": p.name, "group": r.group, "state": r.state,
                "created": r.created_at, "url": url(entity, p.name, r.id), "slurm": m.group(1) if m else None,
                "git_commit": getattr(r, "commit", None), "last_step": r.lastHistoryStep, "tags": list(r.tags), "notes": r.notes,
                "summary": clean(r.summary), "config": clean(r.config),
            })
    runs.sort(key=lambda r: r["created"])
    by_id = {(r["project"], r["id"]): r for r in runs}
    by_id_only = {r["id"]: r for r in runs}
    fetched = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (EVAL / "wandb_runs.json").write_text(json.dumps({"entity": entity, "fetched": fetched, "runs": runs}, indent=1))

    # 2. result -> run checks
    rows = []
    entries = list(rmap["entries"])
    for e in rmap["box_entries"]:
        reg = registry[e["registry_id"]]
        entries.append({
            **e, "wandb_run": reg["wandb_run"], "project": reg["wandb_project"], "corrected_by": f"rescore_{e['registry_id']}",
            "checks": [{"online_key": k, "label": k,
                        "current": {"kind": "rescore", "registry_id": e["registry_id"], "which": "new", "metric": k},
                        "legacy": {"kind": "rescore", "registry_id": e["registry_id"], "which": "legacy", "metric": k},
                        "tol": 1e-3, "expect": "online equals superseded (legacy-scorer) value"} for k in rmap["box_metrics"]],
        })
    entries += rmap.get("audit_entries", [])
    for r in rescore["runs"]:  # every corrected box metric, as logged by the audit run
        entries.append({"id": f"audit_rescore_{r['id']}", "ablation": r["ablation"], "wandb_run": f"rescore_{r['id']}",
                        "what": f"CORRECTED (pycocotools) {r['method']} on {r['dataset']}",
                        "checks": [{"online_key": k, "label": k, "tol": 1e-6,
                                    "current": {"kind": "rescore", "registry_id": r["id"], "which": "new", "metric": k}} for k in rmap["box_metrics"]]})
    for e in entries:
        run = by_id.get((e.get("project", "water-debris-seg"), e["wandb_run"])) or by_id_only.get(e["wandb_run"])
        for c in e["checks"]:
            online = None if run is None else run["summary"].get(c["online_key"])
            cur = value_of(c["current"], rescore)
            leg = value_of(c["legacy"], rescore) if "legacy" in c else None
            v = verdict(online, cur, leg, c.get("tol", 1e-3))
            rows.append({"entry": e["id"], "ablation": e["ablation"], "what": e["what"], "run": e["wandb_run"],
                         "url": run["url"] if run else None, "state": run["state"] if run else "MISSING",
                         "key": c["label"], "online": online, "current": cur, "legacy": leg, "verdict": v,
                         "expect": c.get("expect"), "note": c.get("note"), "corrected": e.get("corrected_by")})

    # 3. which benchmark runs support no cited result
    mapped = {e["wandb_run"] for e in entries}
    roles = {s["run"]: s["role"] for s in rmap.get("supporting_runs", [])}

    def cited(r) -> str:
        if r["id"] in mapped:
            return "yes (value compared in §1)"
        if r["id"] in roles:
            return roles[r["id"]]
        if r["state"] in ("failed", "crashed"):
            return f"{r['state']} attempt (not documented individually)"
        return "no result depends on it"

    unmapped = [r for r in runs if r["id"] not in mapped and r["id"] not in roles and r["state"] not in ("failed", "crashed")]
    attempts = [r for r in runs if r["id"] not in mapped and r["id"] not in roles and r["state"] in ("failed", "crashed")]

    md = ["# W&B audit", "",
          f"Generated {fetched} by `water-debris-seg/scripts/wandb_audit.py` (read-only) from "
          f"[`wandb_result_map.json`](wandb_result_map.json), [`box_eval_registry.json`](box_eval_registry.json) and "
          f"[`RESCORE.json`](RESCORE.json). Every benchmark run with its summary: [`wandb_runs.json`](wandb_runs.json). "
          f"Entity `{entity}`; projects: " + ", ".join(sorted({f"`{r['project'][:40]}…`" if len(r['project']) > 40 else f"`{r['project']}`" for r in runs})) + ".", "",
          "Verdict meaning: **matches current** = online value equals the value now reported; **superseded** = online value is the "
          "pre-2026-10-13 value that was later restated (decision log 2026-10-13); **differs** = see note.", "",
          "## 1. Reported result → W&B run", "",
          "| Abl. | Result | Run | Metric | Online | Now reported | Pre-fix | Verdict |", "|---|---|---|---|---|---|---|---|"]
    corr_link = lambda cid: f"[`{cid}`]({by_id_only[cid]['url']})" if cid in by_id_only else f"`{cid}` (missing)"
    for r in rows:
        link = f"[`{r['run'][:24]}`]({r['url']})" if r["url"] else f"`{r['run']}` (missing)"
        flag = {"matches current": "✅", "online = superseded (legacy) value": "⚠️", "differs": "❌", "not logged": "—"}[r["verdict"]]
        note = f" — {r['note']}" if r["note"] else ""
        if r["corrected"] and r["verdict"] != "matches current":
            note += f" — **corrected value online: {corr_link(r['corrected'])}**"
        elif r["corrected"]:
            note += f" — correction: {corr_link(r['corrected'])}"
        md.append(f"| {r['ablation']} | {r['what']} | {link} | {r['key']} | {fmt(r['online'])} | {fmt(r['current'])} | "
                  f"{fmt(r['legacy'])} | {flag} {r['verdict']}{note} |")
    md += ["", "## 2. Results with no W&B record", ""]
    if rmap["not_in_wandb"]:
        md += ["| What | Slurm / script | Where it lives locally |", "|---|---|---|"]
        md += [f"| {g['what']} | {g['slurm']} | `{g['local']}` |" for g in rmap["not_in_wandb"]]
    else:
        md += ["None remaining. Six results were missing from W&B until 2026-10-13; they were logged by "
               "`wandb_log_corrections.py` in group `audit_2026-10-13` (the originals were not modified):", "",
               "| Previously missing | Now in W&B | Local source |", "|---|---|---|"]
        md += [f"| {g['what']} | `{g['now_in_wandb']}` | `{g['local']}` |" for g in rmap["closed_gaps"]]
    md += ["", "## 3. All benchmark runs on W&B", "", "| Created | Group | Run | State | Slurm | Recorded git commit | Tags | Cited result |", "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        state = r["state"] if r["state"] == "finished" else f"**{r['state']}**"
        c = (r["git_commit"] or "none")[:10]
        md.append(f"| {r['created'][:16]} | {r['group']} | [`{r['name'][:40]}`]({r['url']}) | {state} | {r['slurm'] or '—'} | `{c}` | {', '.join(t for t in r['tags'] if 'superseded' in t or t == 'val-split-only') or '—'} | {cited(r)} |")
    audit_commits = sorted({(r["git_commit"] or "none")[:10] for r in runs if r["group"] == "audit_2026-10-13"})
    md += ["", "## 4. Limitations of what W&B can prove", "",
           "- **Code version (original runs):** the commit W&B recorded is the upstream repo's (`0f1e1ccb` = tue-mps/eomt, `f351ccbf` = tue-mps/videomt), "
           "this project's scaffold commit `5983e7f` (which contains no code), or none. This project's own code was untracked when those runs "
           "executed, so no original run pins it. Commit `9bc24c7` contains the code as of 2026-10-09. "
           f"The correction runs (group `audit_2026-10-13`) pin " + ", ".join(f"`{c}`" for c in audit_commits) + ", which contains the code that produced them.",
           "- **Project names:** Ultralytics sets the W&B project from its `project=` path, so the original YOLO runs sit in path-named projects; the correction runs are all in `water-debris-seg`.",
           "- **Original mask→box runs keep their pre-fix values** (tagged `superseded-scorer-2026-10-13`, with a note linking the correction); values were not edited, so history stays auditable.",
           f"- **{len(mapped)} of {len(runs)} runs** have a value compared in §1; {len(roles)} more back a documented result or invalid run without a comparable logged value; "
           f"{len(attempts)} are failed or crashed attempts; {len(unmapped)} support no result."]
    (EVAL / "WANDB_AUDIT.md").write_text("\n".join(md) + "\n")

    # 4. clickable W&B block inside each ablation README (between markers, regenerated each run)
    by_run_rows: dict[str, list[dict]] = {}
    for r in rows:
        by_run_rows.setdefault(r["run"], []).append(r)

    def audit_cell(r) -> str:
        rs = by_run_rows.get(r["id"])
        if rs:
            vs = {x["verdict"] for x in rs}
            corr = next((x["corrected"] for x in rs if x["corrected"]), None)
            link = f" → corrected: [`{corr}`]({by_id_only[corr]['url']})" if corr in by_id_only else ""
            if "differs" in vs:
                return "❌ online value is a different split than the reported result" + link
            if "online = superseded (legacy) value" in vs:
                return "⚠️ online holds pre-fix (superseded) values" + link
            return "✅ online values match the reported results"
        return roles.get(r["id"]) or (f"{r['state']} attempt" if r["state"] in ("failed", "crashed") else "—")

    start, end = "<!-- wandb-links:start (generated by scripts/wandb_audit.py; do not edit) -->", "<!-- wandb-links:end -->"
    for nn in sorted({m.group(1) for r in runs if (m := re.match(r"^ablation(\d\d)_", r["group"] or ""))}):
        readmes = sorted((EVAL / "ablations").glob(f"{nn}_*/README.md"))
        if not readmes:
            continue
        audit_ids = {e["wandb_run"] for e in entries if e["ablation"] == nn and e["wandb_run"].startswith(("rescore_", "ab0"))
                     and (by_id_only.get(e["wandb_run"]) or {}).get("group") == "audit_2026-10-13"}
        mine = [r for r in runs if (r["group"] or "").startswith(f"ablation{nn}_") or r["id"] in audit_ids]
        block = [start, "", "## W&B runs", "",
                 f"Online runs for this ablation (fetched {fetched[:10]}); value-by-value comparison with the reported results: "
                 "[`WANDB_AUDIT.md`](../../WANDB_AUDIT.md).", "",
                 "| Run | State | Slurm | Audit |", "|---|---|---|---|"]
        block += [f"| [`{r['name'][:48]}`]({r['url']}) | {r['state']} | {r['slurm'] or '—'} | {audit_cell(r)} |" for r in mine]
        block += ["", end]
        text = readmes[0].read_text()
        if start in text:
            text = text[: text.index(start)] + "\n".join(block) + text[text.index(end) + len(end):]
        else:
            text = text.rstrip("\n") + "\n\n" + "\n".join(block) + "\n"
        readmes[0].write_text(text)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(json.dumps({"runs": len(runs), "checks": len(rows), "verdicts": counts, "unmapped_runs": len(unmapped), "failed_attempts": len(attempts)}, indent=1))


if __name__ == "__main__":
    main()
