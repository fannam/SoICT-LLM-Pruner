from __future__ import annotations

from .base import BaseVisionAdapter

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


class Qwen2_5_VLVisionAdapter(BaseVisionAdapter):
    def __init__(self):
        super().__init__(name="qwen2_5_vl", model_cls=Qwen2_5_VLForConditionalGeneration)
