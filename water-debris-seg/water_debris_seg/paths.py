"""Filesystem layout for Snellius (home repo + scratch data)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return _REPO_ROOT


def scratch_root() -> Path:
    return Path(
        os.environ.get(
            "WATER_DEBRIS_SCRATCH_ROOT",
            f"/scratch-shared/{os.environ.get('USER', 'unknown')}/water-debris-seg",
        )
    )


def data_root() -> Path:
    return Path(os.environ.get("WATER_DEBRIS_DATA_ROOT", scratch_root() / "data"))


def third_party_root() -> Path:
    return Path(os.environ.get("WATER_DEBRIS_THIRD_PARTY", scratch_root() / "third_party"))


def dataset_dir(name: str) -> Path:
    return data_root() / name


def eomt_root() -> Path:
    return third_party_root() / "eomt"


def videomt_root() -> Path:
    return third_party_root() / "videomt"


def ultralytics_root() -> Path:
    return third_party_root() / "ultralytics"
