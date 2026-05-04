from __future__ import annotations

import importlib

Qwen3ModelAdapter = None
try:
    Qwen3ModelAdapter = getattr(importlib.import_module(".models.qwen3", __package__), "Qwen3ModelAdapter")
except (ImportError, AttributeError):
    pass

__all__ = []

if Qwen3ModelAdapter is not None:
    __all__.append("Qwen3ModelAdapter")
