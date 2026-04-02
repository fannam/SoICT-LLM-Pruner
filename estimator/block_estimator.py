from __future__ import annotations

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from ._shared import _BaseBlockPerplexityEstimator, _BaseSimilarityBlockEstimator


class Llama3SimilarityBlockEstimator(_BaseSimilarityBlockEstimator):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2SimilarityBlockEstimator(_BaseSimilarityBlockEstimator):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


class Llama3BlockPerplexityEstimator(_BaseBlockPerplexityEstimator):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2BlockPerplexityEstimator(_BaseBlockPerplexityEstimator):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


__all__ = [
    "Llama3SimilarityBlockEstimator",
    "Qwen2SimilarityBlockEstimator",
    "Llama3BlockPerplexityEstimator",
    "Qwen2BlockPerplexityEstimator",
]
