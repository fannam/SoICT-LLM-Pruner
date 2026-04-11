from __future__ import annotations

from transformers import Qwen2ForCausalLM

from .decoder import DecoderModelAdapter


class Qwen2ModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="qwen2", model_cls=Qwen2ForCausalLM)
