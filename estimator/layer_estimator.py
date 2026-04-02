from __future__ import annotations

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from ._shared import _BaseSimilarityLayerEstimator


class Llama3SimilarityLayerEstimator(_BaseSimilarityLayerEstimator):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2SimilarityLayerEstimator(_BaseSimilarityLayerEstimator):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


__all__ = [
    "Llama3SimilarityLayerEstimator",
    "Qwen2SimilarityLayerEstimator",
]
