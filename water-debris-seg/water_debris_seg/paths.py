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


# Datasets moved to home after the 2026-10-23 scratch purge deleted re-created files (decision log).
HOME_DATA = Path(__file__).resolve().parents[2] / "data" / "water_debris"
_ON_HOME = {"cansurf": HOME_DATA / "cansurf_home"}


def dataset_dir(name: str) -> Path:
    if name in _ON_HOME and "WATER_DEBRIS_DATA_ROOT" not in os.environ:
        return _ON_HOME[name]
    return data_root() / name


def eomt_root() -> Path:
    return third_party_root() / "eomt"


def videomt_root() -> Path:
    return third_party_root() / "videomt"


def ultralytics_root() -> Path:
    return third_party_root() / "ultralytics"
