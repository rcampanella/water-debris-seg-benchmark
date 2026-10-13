"""Guard: the LaRS→COCO bridge must point at the COCO classes its comments name (decision log 2026-10-13)."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from water_debris_seg.taxonomy import LARS_TO_COCO_PANOPTIC_CATEGORY  # noqa: E402


def test_bridge_ids_match_commented_coco_names():
    names = {c["id"]: c["name"] for c in json.loads((ROOT / "configs/panoptic_coco_categories.json").read_text())}
    src = (ROOT / "water_debris_seg/taxonomy.py").read_text()
    for lars_id, coco_id in LARS_TO_COCO_PANOPTIC_CATEGORY.items():
        m = re.search(rf"^\s+{lars_id}: {coco_id},\s+# .*-> ([\w-]+)", src, re.M)
        assert m, f"no '-> name' comment for LaRS {lars_id}"
        assert names[coco_id] == m.group(1), f"LaRS {lars_id}: id {coco_id} is {names[coco_id]!r}, comment says {m.group(1)!r}"
