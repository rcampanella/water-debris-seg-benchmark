"""LoRA for Ultralytics YOLO convolutions (ablation 10, LoRA checkpoint).

Ultralytics has no LoRA support. This module adds it without patching Ultralytics:

* ``LoRAConv2d`` replaces the ``nn.Conv2d`` inside every Ultralytics ``Conv`` block of the
  backbone and neck. The pretrained kernel is kept as a *buffer* (so Ultralytics' trainer, which
  re-enables ``requires_grad`` on every frozen floating-point Parameter, cannot unfreeze it) and the
  update is a trainable low-rank factorisation of the flattened kernel:
  ``W = W0 + (alpha / r) * (B @ A).view_as(W0)`` with ``A`` (r x in*kh*kw) Kaiming-initialised
  and ``B`` (out x r) zero-initialised, so training starts exactly at the pretrained model.
* The detection head (last module, ``Detect``) is left untouched and trained fully, because the
  class layer changes from 80 COCO classes to the LaRS classes. BatchNorm layers stay trainable.
* Ultralytics fuses Conv+BN for inference by *assigning* new ``weight``/``bias`` onto the conv
  (``fuse_conv_and_bn``). ``LoRAConv2d`` intercepts that assignment: the getter returns the merged
  kernel, the setter stores the fused kernel and drops the LoRA factors, so fused inference is exact.
* ``merge_lora`` turns a LoRA model into a plain Ultralytics model (``nn.Conv2d`` with merged
  kernels) that loads without this module.

Checkpoints saved by the trainer pickle ``LoRAConv2d``; import this module (or have
``water-debris-seg`` on ``PYTHONPATH``) before loading them, or use the merged checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """Frozen conv kernel (buffer) + trainable low-rank kernel update."""

    def __init__(self, conv: nn.Conv2d, r: int, alpha: float):
        super().__init__()
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size, self.stride, self.padding = conv.kernel_size, conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.padding_mode = conv.padding_mode
        self.r = r
        self.scale = alpha / r
        self.register_buffer("base_weight", conv.weight.detach().clone())
        self.register_buffer("base_bias", None if conv.bias is None else conv.bias.detach().clone())
        fan_in = (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        self.lora_A = nn.Parameter(torch.empty(r, fan_in, device=conv.weight.device, dtype=conv.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, r, device=conv.weight.device, dtype=conv.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    # --- the kernel Ultralytics (and fuse_conv_and_bn) see -------------------------------------
    @property
    def weight(self) -> torch.Tensor:
        if self.merged:
            return self.base_weight
        # Always fp32 with autocast off: under AMP an fp16 B@A overflows in backward (the kernel gradient,
        # already multiplied by the loss scale, is summed over in*kh*kw terms), producing NaN gradients that
        # make GradScaler skip every optimizer step. forward() casts the merged kernel to the input dtype,
        # exactly as AMP does for an ordinary conv weight.
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            delta = (self.lora_B.float() @ self.lora_A.float()).view(self.base_weight.shape)
            return self.base_weight.float() + self.scale * delta

    @property
    def bias(self):
        return self.base_bias

    def __setattr__(self, name, value):
        # fuse_conv_and_bn assigns the fused kernel/bias: store them and drop the LoRA factors
        if name == "weight":
            self.base_weight = value.detach()
            self.merged = True
            for p in ("lora_A", "lora_B"):
                if p in self._parameters:
                    del self._parameters[p]
            return
        if name == "bias":
            self.base_bias = None if value is None else value.detach()
            return
        super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        return F.conv2d(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype),
                        self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, k={self.kernel_size}, s={self.stride}, r={self.r}, merged={self.merged}"


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> dict:
    """Wrap the conv of every Ultralytics Conv block outside the Detect head. Returns parameter counts."""
    from ultralytics.nn.modules import Conv

    layers = model.model  # DetectionModel: nn.Sequential, last entry is Detect
    n = 0
    for layer in list(layers)[:-1]:
        for m in layer.modules():
            if isinstance(m, Conv) and isinstance(m.conv, nn.Conv2d):
                m.conv = LoRAConv2d(m.conv, r, alpha)
                n += 1
    total = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers() if b is not None)
    lora = sum(p.numel() for n_, p in model.named_parameters() if "lora_" in n_)
    trainable = sum(p.numel() for p in model.parameters())
    return {"wrapped_convs": n, "lora_params": lora, "trainable_params": trainable, "total_params_incl_frozen": total}


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRAConv2d by an nn.Conv2d holding the merged kernel (in place)."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRAConv2d):
                conv = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size, child.stride, child.padding,
                                 child.dilation, child.groups, bias=child.bias is not None, padding_mode=child.padding_mode)
                with torch.no_grad():
                    conv.weight.copy_(child.weight.detach())
                    if child.bias is not None:
                        conv.bias.copy_(child.bias)
                conv = conv.to(child.base_weight.device, child.base_weight.dtype)
                setattr(parent, name, conv)
    return model

