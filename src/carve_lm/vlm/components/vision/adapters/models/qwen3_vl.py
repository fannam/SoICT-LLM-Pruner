from __future__ import annotations

from ..base import BaseVisionAdapter, VisionMLPProjectionBundle

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


class Qwen3VLVisionAdapter(BaseVisionAdapter):
    def __init__(self):
        super().__init__(name="qwen3_vl", model_cls=Qwen3VLForConditionalGeneration)

    def get_mlp_projections(self, block) -> VisionMLPProjectionBundle:
        mlp = self.get_mlp_module(block)
        return VisionMLPProjectionBundle(
            linear_fc1=mlp.linear_fc1,
            linear_fc2=mlp.linear_fc2,
        )
