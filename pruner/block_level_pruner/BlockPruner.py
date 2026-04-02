from __future__ import annotations

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from .._shared import _BaseBlockPruner


class Llama3BlockPruner(_BaseBlockPruner):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2BlockPruner(_BaseBlockPruner):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


__all__ = [
    "Llama3BlockPruner",
    "Qwen2BlockPruner",
]
