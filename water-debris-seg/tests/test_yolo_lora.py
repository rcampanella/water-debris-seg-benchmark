"""Guards for water_debris_seg/yolo_lora.py: LoRA must be exact at init, through Conv+BN fusion and after merging."""

import copy
import io
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ultralytics = pytest.importorskip("ultralytics")
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from water_debris_seg.yolo_lora import LoRAConv2d, inject_lora, merge_lora  # noqa: E402


def _model():
    torch.manual_seed(0)
    m = DetectionModel("yolo11n.yaml", nc=9, verbose=False).eval()
    return m


def _out(m, x):
    with torch.no_grad():
        y = m(x)
    return y[0] if isinstance(y, (list, tuple)) else y


def _randomise_lora(m):
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, LoRAConv2d):
                mod.lora_B.normal_(0, 0.02)


def test_lora_is_identity_at_init_and_freezes_base():
    x = torch.rand(1, 3, 128, 128)
    base = _model()
    lora = copy.deepcopy(base)
    info = inject_lora(lora, r=4, alpha=8)
    assert info["wrapped_convs"] > 0
    assert torch.allclose(_out(base, x), _out(lora, x), atol=1e-5)
    names = [n for n, _ in lora.named_parameters()]
    assert not any(n.endswith("conv.weight") and "model.23" not in n for n in names), "backbone kernels must not be Parameters"
    assert info["trainable_params"] < sum(p.numel() for p in base.parameters())


def test_fused_and_merged_match_unfused():
    x = torch.rand(1, 3, 128, 128)
    m = _model()
    inject_lora(m, r=4, alpha=8)
    _randomise_lora(m)
    ref = _out(m, x)
    fused = copy.deepcopy(m).fuse(verbose=False)
    assert torch.allclose(ref, _out(fused, x), atol=1e-4)
    merged = merge_lora(copy.deepcopy(m))
    assert not any(isinstance(mod, LoRAConv2d) for mod in merged.modules())
    assert torch.allclose(ref, _out(merged, x), atol=1e-5)


def test_pickle_round_trip():
    m = _model()
    inject_lora(m, r=4, alpha=8)
    _randomise_lora(m)
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    m2 = torch.load(buf, weights_only=False)
    x = torch.rand(1, 3, 96, 96)
    assert torch.allclose(_out(m, x), _out(m2, x), atol=1e-6)


def test_gradients_reach_lora_not_base():
    m = _model().train()
    inject_lora(m, r=4, alpha=8)
    _randomise_lora(m)                      # with B = 0 only B gets gradient; randomised B lets A train too
    y = m(torch.rand(2, 3, 96, 96))

    def tensors(o):  # train-mode output may be a tensor, list/tuple or dict depending on the Ultralytics version
        if torch.is_tensor(o):
            yield o
        elif isinstance(o, dict):
            for v in o.values():
                yield from tensors(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from tensors(v)

    sum(t.float().sum() for t in tensors(y)).backward()
    loras = [mod for mod in m.modules() if isinstance(mod, LoRAConv2d)]
    assert all(mod.lora_B.grad is not None and mod.lora_B.grad.abs().sum() > 0 for mod in loras)
    assert all(mod.lora_A.grad is not None and mod.lora_A.grad.abs().sum() > 0 for mod in loras)
    assert all(not mod.base_weight.requires_grad for mod in loras)


def test_lora_kernel_is_fp32_under_autocast():
    """Regression: an fp16 B@A under AMP overflowed in backward -> NaN grads -> every optimizer step skipped."""
    m = _model()
    inject_lora(m, r=4, alpha=8)
    mod = next(x for x in m.modules() if isinstance(x, LoRAConv2d))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert mod.weight.dtype == torch.float32
        y = mod(torch.rand(1, mod.in_channels, 16, 16, dtype=torch.bfloat16))
    y.float().sum().backward()
    assert torch.isfinite(mod.lora_B.grad).all() and mod.lora_B.grad.abs().sum() > 0
