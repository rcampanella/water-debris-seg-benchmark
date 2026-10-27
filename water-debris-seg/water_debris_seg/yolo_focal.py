"""Focal classification loss for Ultralytics YOLO detection (ablation 10, focal variants).

Ultralytics' detection loss (``v8DetectionLoss``) uses plain ``BCEWithLogitsLoss`` for the class
scores; its ``FocalLoss`` class is not wired into detection. This module adds focal loss without
patching Ultralytics:

* ``FocalBCE`` is an element-wise drop-in for ``v8DetectionLoss.bce``:
  ``FL = alpha_t * (1 - p_t) ** gamma * BCE(logit, target)`` with
  ``p_t = t * p + (1 - t) * (1 - p)`` and ``alpha_t = t * alpha + (1 - t) * (1 - alpha)``.
  Targets are YOLO's soft task-aligned scores (0..1); the formulas take them as is, as YOLOv5's
  ``FocalLoss`` (``fl_gamma``) did. The loss is still summed and divided by the target-score sum.
* ``FocalDetectionModel`` is a ``DetectionModel`` whose criterion uses ``FocalBCE``. Training swaps
  the model's class to it (``use_focal``), so the EMA copy, the validation loss and a resumed run all
  use the same loss. Checkpoints pickle this class: import this module before loading them.
  ``to_plain_detection_model`` turns it back into a plain ``DetectionModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel


class FocalBCE(nn.Module):
    """Element-wise focal loss on logits (reduction='none'), soft targets allowed."""

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = pred.sigmoid()
        p_t = target * p + (1 - target) * (1 - p)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        return alpha_t * (1.0 - p_t).pow(self.gamma) * bce


class FocalDetectionModel(DetectionModel):
    """DetectionModel whose classification loss is FocalBCE (gamma/alpha stored on the model)."""

    def init_criterion(self):
        crit = super().init_criterion()
        if not hasattr(crit, "bce"):
            raise TypeError(f"focal loss expects a criterion with a .bce term, got {type(crit).__name__}")
        crit.bce = FocalBCE(getattr(self, "focal_gamma", 1.5), getattr(self, "focal_alpha", 0.25))
        return crit


def use_focal(model: DetectionModel, gamma: float = 1.5, alpha: float = 0.25) -> DetectionModel:
    """Switch a DetectionModel (in place) to focal classification loss."""
    if type(model) is not DetectionModel:
        raise TypeError(f"expected a plain DetectionModel, got {type(model).__name__}")
    model.__class__ = FocalDetectionModel
    model.focal_gamma, model.focal_alpha = float(gamma), float(alpha)
    model.criterion = None
    return model


def to_plain_detection_model(model: nn.Module) -> nn.Module:
    """Undo use_focal (in place) so the checkpoint loads without this module. Inference is unaffected."""
    if isinstance(model, FocalDetectionModel):
        model.__class__ = DetectionModel
    return model
