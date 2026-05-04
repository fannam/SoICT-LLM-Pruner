from __future__ import annotations

import importlib
from typing import Any

from .base import BaseMergerAdapter, MergerProjectionBundle

__all__ = [
    "BaseMergerAdapter",
    "MergerProjectionBundle",
    "get_model_adapter",
    "register_model_adapter",
    "registered_model_adapters",
    "resolve_model_adapter",
]

_MODEL_ADAPTERS: list[Any] = []


def register_model_adapter(adapter):
    name = getattr(adapter, "name", None)
    if name is None:
        raise TypeError("Merger adapters must define a `name` attribute.")
    if any(getattr(existing, "name", None) == name for existing in _MODEL_ADAPTERS):
        raise KeyError("Model adapter '{}' is already registered.".format(name))
    _MODEL_ADAPTERS.append(adapter)
    return adapter


def get_model_adapter(name: str):
    for adapter in _MODEL_ADAPTERS:
        if getattr(adapter, "name", None) == name:
            return adapter
    available = ", ".join(getattr(adapter, "name", "<unnamed>") for adapter in _MODEL_ADAPTERS) or "<empty>"
    raise KeyError("Unknown model adapter '{}'. Available: {}.".format(name, available))


def registered_model_adapters() -> tuple[Any, ...]:
    return tuple(_MODEL_ADAPTERS)


def resolve_model_adapter(model, adapter=None):
    if adapter is None:
        for candidate in _MODEL_ADAPTERS:
            matches = getattr(candidate, "matches", None)
            if callable(matches) and matches(model):
                return candidate
        available = ", ".join(getattr(candidate, "name", "<unnamed>") for candidate in _MODEL_ADAPTERS) or "<empty>"
        raise TypeError(
            "No registered merger adapter matches model type {}. Registered adapters: {}.".format(
                type(model).__name__,
                available,
            )
        )

    resolved = get_model_adapter(adapter) if isinstance(adapter, str) else adapter
    ensure_supported = getattr(resolved, "ensure_supported", None)
    if callable(ensure_supported):
        ensure_supported(model)
    return resolved


try:
    Qwen2_5_VLMergerAdapter = getattr(
        importlib.import_module(".qwen2_5_vl", __name__),
        "Qwen2_5_VLMergerAdapter",
    )
    register_model_adapter(Qwen2_5_VLMergerAdapter())
    __all__.append("Qwen2_5_VLMergerAdapter")
except (ImportError, AttributeError, KeyError):
    Qwen2_5_VLMergerAdapter = None
