from __future__ import annotations

from transformers import MistralForCausalLM

from .decoder import DecoderModelAdapter


class MistralModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="mistral", model_cls=MistralForCausalLM)
