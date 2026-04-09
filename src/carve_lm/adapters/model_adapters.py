"""
Backward-compatibility shim. Import directly from the sub-modules instead:
  - base.py       – dataclasses, BaseModelAdapter, utilities
  - decoder.py    – DecoderModelAdapter, GenericDecoderModelAdapter
  - llama.py      – LlamaModelAdapter
  - qwen2.py      – Qwen2ModelAdapter
  - mistral.py    – MistralModelAdapter
  - registry.py   – register_model_adapter, get_model_adapter, resolve_model_adapter, …
"""
from __future__ import annotations

from .base import (
    AttentionHandles,
    AttentionProjectionBundle,
    BaseModelAdapter,
    LayerHandles,
    MLPHandles,
    MLPProjectionBundle,
    _is_json_like,
    import_object,
    object_path,
)
from .decoder import DecoderModelAdapter, GenericDecoderModelAdapter
from .llama import LlamaModelAdapter
from .mistral import MistralModelAdapter
from .qwen2 import Qwen2ModelAdapter
from .registry import (
    get_model_adapter,
    register_model_adapter,
    registered_model_adapters,
    resolve_model_adapter,
)

__all__ = [
    "AttentionHandles",
    "AttentionProjectionBundle",
    "BaseModelAdapter",
    "DecoderModelAdapter",
    "GenericDecoderModelAdapter",
    "LayerHandles",
    "LlamaModelAdapter",
    "MLPHandles",
    "MLPProjectionBundle",
    "MistralModelAdapter",
    "Qwen2ModelAdapter",
    "_is_json_like",
    "get_model_adapter",
    "import_object",
    "object_path",
    "register_model_adapter",
    "registered_model_adapters",
    "resolve_model_adapter",
]
