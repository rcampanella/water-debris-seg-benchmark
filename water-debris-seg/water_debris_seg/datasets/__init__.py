"""Dataset adapters (LaRS first; box sets as downloads land)."""

from water_debris_seg.datasets.cansurf import CansurfLayout, verify_cansurf_layout
from water_debris_seg.datasets.lars import LarsLayout, verify_lars_layout

__all__ = [
    "CansurfLayout",
    "LarsLayout",
    "verify_cansurf_layout",
    "verify_lars_layout",
]
