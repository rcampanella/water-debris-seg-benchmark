"""Guards for water_debris_seg/yolo_focal.py: the focal term, its wiring into Ultralytics' loss, and checkpoints."""

import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ultralytics = pytest.importorskip("ultralytics")
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from water_debris_seg.yolo_focal import FocalBCE, FocalDetectionModel, to_plain_detection_model, use_focal  # noqa: E402
from water_debris_seg.yolo_lora import inject_lora  # noqa: E402


def _model():
    torch.manual_seed(0)
    m = DetectionModel("yolo11n.yaml", nc=9, verbose=False)
    m.args = get_cfg()
    return m


def _batch():
    return {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0.0, 1.0]), "cls": torch.tensor([[7.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.6, 0.1, 0.3]])}


def test_focal_formula():
    x, t = torch.randn(4, 9), torch.rand(4, 9)
    assert torch.allclose(FocalBCE(gamma=0.0, alpha=0.5)(x, t), 0.5 * F.binary_cross_entropy_with_logits(x, t, reduction="none"))
    p = x.sigmoid()
    p_t = t * p + (1 - t) * (1 - p)
    ref = (t * 0.25 + (1 - t) * 0.75) * (1 - p_t) ** 1.5 * F.binary_cross_entropy_with_logits(x, t, reduction="none")
    assert torch.allclose(FocalBCE()(x, t), ref)
    # easy negatives are down-weighted far more than hard ones
    easy, hard = FocalBCE()(torch.tensor([-6.0]), torch.tensor([0.0])), FocalBCE()(torch.tensor([2.0]), torch.tensor([0.0]))
    assert (easy / F.binary_cross_entropy_with_logits(torch.tensor([-6.0]), torch.tensor([0.0]))) < 1e-3 < hard


def test_criterion_uses_focal_and_trains():
    m = use_focal(_model().train(), gamma=1.5, alpha=0.25)
    loss, items = m.loss(_batch())
    assert isinstance(m.criterion.bce, FocalBCE) and m.criterion.bce.gamma == 1.5
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert m.model[-1].cv3[0][-1].weight.grad.abs().sum() > 0      # class branch receives gradient
    base = _model().train()
    assert m.loss(_batch())[0][1] < base.loss(_batch())[0][1]      # focal cls loss is smaller than BCE at init


def test_focal_with_lora():
    m = _model().train()
    use_focal(m)
    inject_lora(m, r=4, alpha=8)
    loss, _ = m.loss(_batch())
    loss.sum().backward()
    assert isinstance(m.criterion.bce, FocalBCE)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "lora_A" in n or "lora_B" in n)


def test_pickle_and_plain_conversion():
    m = use_focal(_model(), gamma=2.0, alpha=0.5).eval()
    m.criterion = None                                             # what the trainer does before saving
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    assert isinstance(m2, FocalDetectionModel) and m2.focal_gamma == 2.0
    x = torch.rand(1, 3, 96, 96)
    with torch.no_grad():
        y1 = m(x)[0]
        y2 = to_plain_detection_model(m2)(x)[0]
    assert type(m2) is DetectionModel and torch.allclose(y1, y2)
