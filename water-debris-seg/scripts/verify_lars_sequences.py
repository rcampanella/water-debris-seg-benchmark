#!/usr/bin/env python3
"""Check LaRS v1.0.0 sequence images are extracted on scratch."""

from __future__ import annotations

from pathlib import Path

from water_debris_seg.paths import scratch_root


def main() -> None:
    root = scratch_root() / "data/lars"
    marker = root / "lars_v1.0.0_images_seq.extracted"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing {marker}; run download_lars_sequences.sh")
    seq_dirs = [p for p in root.iterdir() if p.is_dir() and "seq" in p.name.lower()]
    if not seq_dirs:
        # common layout: train/val may contain sequence subfolders after unzip
        n = sum(1 for _ in root.rglob("*.jpg"))
        if n < 1000:
            raise FileNotFoundError(f"Expected sequence JPGs under {root}, found {n}")
        print(f"OK: {n} jpg files under {root} (sequences extracted)")
        return
    print(f"OK: sequence dirs {seq_dirs[:3]}")


if __name__ == "__main__":
    main()
