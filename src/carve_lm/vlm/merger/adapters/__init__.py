from __future__ import annotations

from typing import Any

__all__ = [
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
