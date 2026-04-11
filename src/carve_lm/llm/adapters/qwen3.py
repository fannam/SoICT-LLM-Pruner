from __future__ import annotations

from .decoder import DecoderModelAdapter

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None


if Qwen3ForCausalLM is not None:
    class Qwen3ModelAdapter(DecoderModelAdapter):
        """Adapter for Qwen3ForCausalLM.

        Qwen3Attention adds per-head-dim q_norm and k_norm (shape: head_dim,) on top of
        the standard decoder layout. These norms are applied after the projection along the
        head_dim axis and are therefore independent of both the head count and the model
        hidden_size, so no special handling is required during pruning.
        """

        def __init__(self):
            super().__init__(name="qwen3", model_cls=Qwen3ForCausalLM)
