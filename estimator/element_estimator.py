from __future__ import annotations

from transformers import LlamaForCausalLM, Qwen2ForCausalLM

from ._shared import _BaseActivationElementEstimator, _BaseWeightMagnitudeEstimator


class Llama3ActivationElementEstimator(_BaseActivationElementEstimator):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2ActivationElementEstimator(_BaseActivationElementEstimator):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


class Llama3WeightMagnitudeEstimator(_BaseWeightMagnitudeEstimator):
    model_cls = LlamaForCausalLM
    model_name = "LlamaForCausalLM"


class Qwen2WeightMagnitudeEstimator(_BaseWeightMagnitudeEstimator):
    model_cls = Qwen2ForCausalLM
    model_name = "Qwen2ForCausalLM"


__all__ = [
    "Llama3ActivationElementEstimator",
    "Qwen2ActivationElementEstimator",
    "Llama3WeightMagnitudeEstimator",
    "Qwen2WeightMagnitudeEstimator",
]
