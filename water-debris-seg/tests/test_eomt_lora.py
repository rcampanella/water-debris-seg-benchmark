"""Guards for water_debris_seg/eomt_lora.py: exact at init, frozen pretrained weights, gradients, merge, annealing."""

import copy
import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
timm = pytest.importorskip("timm")
from water_debris_seg.eomt_lora import LoRALinear, annealing_steps, inject_lora, merge_lora  # noqa: E402


class Net(nn.Module):
    """Same attribute layout as EoMT: network.encoder.backbone (timm ViT) + parts outside the encoder."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.encoder = nn.Module()
        self.encoder.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, reg_tokens=4,
                                                  img_size=64)
        self.q = nn.Embedding(5, self.encoder.backbone.embed_dim)       # stands in for EoMT's queries
        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, 4)

    def forward(self, x):
        f = self.encoder.backbone.forward_features(x)
        return self.class_head(f.mean(1) + self.q.weight.mean(0))


def _x():
    return torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))   # same input on every call


def test_identity_at_init_and_frozen_encoder():
    base = Net().eval()
    lora = copy.deepcopy(base)
    info = inject_lora(lora, r=4, alpha=8)
    assert info["wrapped_linears"] == 4 * len(base.encoder.backbone.blocks)
    with torch.no_grad():
        assert torch.allclose(base(_x()), lora(_x()), atol=1e-6)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert all(("lora_" in n) or ("norm" in n) or not n.startswith("encoder.") for n in trainable)
    assert {"q.weight", "class_head.weight"} <= trainable
    assert not any(n.endswith("qkv.weight") for n, _ in lora.named_parameters()), "pretrained linears must be buffers"
    assert not lora.encoder.backbone.patch_embed.proj.weight.requires_grad


def test_lora_params_sit_under_blocks_for_eomt_lr_decay():
    # EoMT's configure_optimizers assigns layer-wise lr by the "blocks.<i>" part of backbone parameter names
    net = Net()
    inject_lora(net, r=4, alpha=8)
    names = [n for n, _ in net.encoder.backbone.named_parameters() if "lora_" in n]
    assert names and all(n.startswith("blocks.") and n.split(".")[1].isdigit() for n in names)


def test_gradients_reach_lora_only():
    net = Net().train()
    inject_lora(net, r=4, alpha=8)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.02)
    net(_x()).sum().backward()
    loras = [m for m in net.modules() if isinstance(m, LoRALinear)]
    assert all(m.lora_A.grad is not None and m.lora_B.grad.abs().sum() > 0 for m in loras)
    assert all(not m.weight.requires_grad for m in loras)
    assert net.encoder.backbone.patch_embed.proj.weight.grad is None


def test_merge_matches_and_state_dict_round_trip():
    net = Net().eval()
    inject_lora(net, r=4, alpha=8)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.02)
        ref = net(_x())
        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        buf.seek(0)
        net2 = Net().eval()
        inject_lora(net2, r=4, alpha=8)
        net2.load_state_dict(torch.load(buf))
        assert torch.allclose(ref, net2(_x()), atol=1e-6)
        merged = merge_lora(copy.deepcopy(net))
        assert not any(isinstance(m, LoRALinear) for m in merged.modules())
        assert torch.allclose(ref, merged(_x()), atol=1e-5)


def test_annealing_scaled_to_run_length():
    s, e = annealing_steps(7812)             # ablation 03's run length
    assert s == [1302, 3255, 5208] and e == [2604, 4557, 6510]
    assert all(0 < a < b <= 7812 for a, b in zip(s, e))
    s, e = annealing_steps(88692)            # COCO 12 epochs (14782 x 6 = 88692): reproduces EoMT's recipe within rounding
    assert all(abs(a - b) <= 10 for a, b in zip(s + e, [14782, 36955, 59128, 29564, 51737, 73910]))
