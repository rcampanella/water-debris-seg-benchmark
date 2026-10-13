#!/usr/bin/env python3
"""Quick checks for VidEoMT / Detectron2 stack on Snellius (no training)."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def check_import(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001
        return f"FAIL: {exc}"


def main() -> int:
    scratch = Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")
    vid = scratch / "third_party/videomt"
    seq_marker = scratch / "data/lars/lars_v1.0.0_images_seq.extracted"

    print("=== VidEoMT env smoke ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {check_import('torch')}")
    print(f"torchvision: {check_import('torchvision')}")
    print(f"detectron2: {check_import('detectron2')}")
    print(f"videomt repo: {'OK' if vid.is_dir() else 'MISSING'} ({vid})")
    print(f"LaRS sequences marker: {'OK' if seq_marker.is_file() else 'MISSING'}")

    if vid.is_dir():
        r = subprocess.run(
            ["git", "-C", str(vid), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            print(f"videomt git: {r.stdout.strip()}")

    d2_ok = not check_import("detectron2").startswith("FAIL")
    if not d2_ok:
        print(
            "\nNext: install detectron2 in a dedicated env (see videomt README); "
            "eomt venv may not include it."
        )
        return 1
    print("\nSmoke passed (imports + paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
