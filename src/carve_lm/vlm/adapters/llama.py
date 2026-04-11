from __future__ import annotations

from transformers import LlamaForCausalLM

from .decoder import DecoderModelAdapter


class LlamaModelAdapter(DecoderModelAdapter):
    def __init__(self):
        super().__init__(name="llama", model_cls=LlamaForCausalLM)
