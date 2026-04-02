from __future__ import annotations

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from .._shared import _BaseLayerPruner


class Llama3LayerPruner(_BaseLayerPruner):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2LayerPruner(_BaseLayerPruner):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


__all__ = [
    "Llama3LayerPruner",
    "Qwen2LayerPruner",
]
