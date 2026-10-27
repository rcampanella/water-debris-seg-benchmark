"""Importable EoMT LightningModules with LoRA (fairness fix 4), for LightningCLI ``class_path`` in configs.

Defined at module level (LightningCLI resolves classes by import path; a class built inside a factory function is not
importable). Needs EoMT (``training``) on PYTHONPATH, so it is kept apart from ``eomt_lora.py``.
"""

from training.mask_classification_instance import MaskClassificationInstance  # type: ignore[import-untyped]
from training.mask_classification_panoptic import MaskClassificationPanoptic  # type: ignore[import-untyped]

from water_debris_seg.eomt_lora import inject_lora


class LoRAMaskClassificationPanoptic(MaskClassificationPanoptic):
    """EoMT panoptic with LoRA injected after the pretrained weights load (ckpt_path is loaded by the parent)."""

    def __init__(self, *args, lora_r: int = 16, lora_alpha: float = 32.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_info = inject_lora(self.network, lora_r, lora_alpha)
        print(f"[LoRA] r={lora_r} alpha={lora_alpha} {self.lora_info}", flush=True)


class LoRAMaskClassificationInstance(MaskClassificationInstance):
    """EoMT instance segmentation with LoRA (fix 3 / 4 on FloW and CANSURF)."""

    def __init__(self, *args, lora_r: int = 16, lora_alpha: float = 32.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_info = inject_lora(self.network, lora_r, lora_alpha)
        print(f"[LoRA] r={lora_r} alpha={lora_alpha} {self.lora_info}", flush=True)
