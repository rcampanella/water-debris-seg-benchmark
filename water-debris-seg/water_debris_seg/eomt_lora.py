"""LoRA for EoMT's ViT encoder, and EoMT attention-mask annealing scaled to the run length (fairness fix 4).

* ``LoRALinear`` replaces an ``nn.Linear`` (attention ``qkv`` / ``proj``, MLP ``fc1`` / ``fc2`` of every encoder block).
  The pretrained weight and bias are *buffers*, so no optimizer ever updates them, and the update is
  ``(alpha / r) * B @ A`` with ``B`` zero-initialised: training starts exactly at the pretrained model.
  EoMT calls these layers as modules (``module.qkv(x)``, ``module.proj(...)``, timm ``Mlp``), so the wrapper is a
  drop-in. LoRA factors live under ``backbone.blocks.<i>``, so EoMT's layer-wise learning-rate decay (which groups
  parameters by that name) gives each factor its block's learning rate without changes to EoMT.
* ``inject_lora`` also freezes the rest of the pretrained encoder (patch / position embeddings, prefix tokens) and leaves
  trainable: LoRA factors, LayerNorms, and everything outside the encoder (queries, class and mask heads).
* ``merge_lora`` folds the update into plain ``nn.Linear`` layers for export.
* ``annealing_steps`` scales EoMT's COCO schedule (masked attention annealed off over epochs 2-4, 5-7, 8-10 of 12,
  i.e. fractions 2/12..4/12, 5/12..7/12, 8/12..10/12) to a run of any length. Ablation 03 copied COCO's absolute steps
  into a run 11x shorter, so its annealing never started.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
COCO_START_FRAC = (2 / 12, 5 / 12, 8 / 12)
COCO_END_FRAC = (4 / 12, 7 / 12, 10 / 12)


class LoRALinear(nn.Module):
    """Frozen linear layer (buffers) + trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.in_features, self.out_features, self.r = base.in_features, base.out_features, r
        # scale saved in the state dict so a checkpoint can be rebuilt without the training config
        self.register_buffer("lora_scale", torch.tensor(float(alpha) / r))
        self.register_buffer("weight", base.weight.detach().clone())
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().clone())
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.lora_scale.to(x.dtype) * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_scale * (self.lora_B.float() @ self.lora_A.float())).to(self.weight.dtype)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, r={self.r}, scale={float(self.lora_scale)}"


def _get(module: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        module = getattr(module, part)
    return module


def inject_lora(network: nn.Module, r: int = 16, alpha: float = 32.0, targets=TARGETS) -> dict:
    """Wrap the target linears of every encoder block; freeze the remaining pretrained encoder weights except LayerNorms."""
    backbone = network.encoder.backbone
    n = 0
    for block in backbone.blocks:
        for t in targets:
            parent_path, _, leaf = t.rpartition(".")
            parent = _get(block, parent_path)
            layer = getattr(parent, leaf)
            if isinstance(layer, nn.Linear):
                setattr(parent, leaf, LoRALinear(layer, r, alpha))
                n += 1
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            continue
        p.requires_grad_("norm" in name)       # LayerNorms (block norms, final norm) stay trainable
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    total = sum(p.numel() for p in network.parameters()) + sum(b.numel() for b in network.buffers())
    lora = sum(p.numel() for n_, p in network.named_parameters() if "lora_" in n_)
    return {"wrapped_linears": n, "lora_params": lora, "trainable_params": trainable, "total_params_incl_frozen": total}


def merge_lora(network: nn.Module) -> nn.Module:
    """Replace every LoRALinear with an nn.Linear holding the merged weight (in place)."""
    for parent in list(network.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                with torch.no_grad():
                    lin.weight.copy_(child.merged_weight())
                    if child.bias is not None:
                        lin.bias.copy_(child.bias)
                setattr(parent, name, lin.to(child.weight.device, child.weight.dtype))
    return network


def annealing_steps(total_steps: int, num_blocks: int = 3) -> tuple[list[int], list[int]]:
    """EoMT's COCO attention-mask annealing schedule, scaled to ``total_steps`` (same fractions of training)."""
    assert num_blocks == len(COCO_START_FRAC), "schedule defined for EoMT's 3 query blocks"
    return ([round(f * total_steps) for f in COCO_START_FRAC], [round(f * total_steps) for f in COCO_END_FRAC])


def lora_lightning_class(base_cls):
    """Subclass of an EoMT LightningModule (panoptic or instance) that injects LoRA after the pretrained weights load.

    Reads r / alpha from the ``lora_r`` / ``lora_alpha`` init arguments, so LightningCLI configs can set them."""

    class LoRAModule(base_cls):
        def __init__(self, *args, lora_r: int = 16, lora_alpha: float = 32.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.lora_info = inject_lora(self.network, lora_r, lora_alpha)
            print(f"[LoRA] r={lora_r} alpha={lora_alpha} {self.lora_info}", flush=True)

    LoRAModule.__name__ = f"LoRA{base_cls.__name__}"
    return LoRAModule
