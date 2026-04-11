"""
Backward-compatibility shim. Import directly from the sub-modules instead:
  - base.py       – dataclasses, BaseModelAdapter, utilities
  - decoder.py    – DecoderModelAdapter, GenericDecoderModelAdapter
  - qwen2_5_vl.py – Qwen2_5_VLModelAdapter (when supported by transformers)
  - registry.py   – register_model_adapter, get_model_adapter, resolve_model_adapter, …
"""
from __future__ import annotations

import importlib

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
from .registry import (
    get_model_adapter,
    register_model_adapter,
    registered_model_adapters,
    resolve_model_adapter,
)

Qwen2_5_VLModelAdapter = None
try:
    Qwen2_5_VLModelAdapter = getattr(
        importlib.import_module(".qwen2_5_vl", __name__.rsplit(".", 1)[0]),
        "Qwen2_5_VLModelAdapter",
    )
except (ImportError, AttributeError):
    pass

__all__ = [
    "AttentionHandles",
    "AttentionProjectionBundle",
    "BaseModelAdapter",
    "DecoderModelAdapter",
    "GenericDecoderModelAdapter",
    "LayerHandles",
    "MLPHandles",
    "MLPProjectionBundle",
    "_is_json_like",
    "get_model_adapter",
    "import_object",
    "object_path",
    "register_model_adapter",
    "registered_model_adapters",
    "resolve_model_adapter",
]

if Qwen2_5_VLModelAdapter is not None:
    __all__.append("Qwen2_5_VLModelAdapter")
