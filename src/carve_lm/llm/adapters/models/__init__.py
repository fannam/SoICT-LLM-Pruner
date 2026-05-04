from __future__ import annotations

import importlib

from .llama import LlamaModelAdapter
from .mistral import MistralModelAdapter
from .qwen2 import Qwen2ModelAdapter

Qwen3ModelAdapter = None
try:
    Qwen3ModelAdapter = getattr(importlib.import_module(".qwen3", __name__), "Qwen3ModelAdapter")
except (ImportError, AttributeError):
    pass

__all__ = [
    "LlamaModelAdapter",
    "MistralModelAdapter",
    "Qwen2ModelAdapter",
]

if Qwen3ModelAdapter is not None:
    __all__.append("Qwen3ModelAdapter")
