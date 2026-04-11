from __future__ import annotations

import importlib

from transformers import PreTrainedModel

from .base import BaseModelAdapter
from .decoder import GenericDecoderModelAdapter
from .llama import LlamaModelAdapter
from .mistral import MistralModelAdapter
from .qwen2 import Qwen2ModelAdapter

_MODEL_ADAPTERS: list[BaseModelAdapter] = []
_FALLBACK_MODEL_ADAPTER = GenericDecoderModelAdapter()


def _load_optional_adapter(module_name: str, class_name: str):
    try:
        module = importlib.import_module(".{}".format(module_name), __package__)
    except ImportError:
        return None
    return getattr(module, class_name, None)


def register_model_adapter(adapter: BaseModelAdapter) -> BaseModelAdapter:
    if any(existing.name == adapter.name for existing in _MODEL_ADAPTERS):
        raise KeyError("Model adapter '{}' is already registered.".format(adapter.name))
    _MODEL_ADAPTERS.append(adapter)
    return adapter


def get_model_adapter(name: str) -> BaseModelAdapter:
    for adapter in _MODEL_ADAPTERS:
        if adapter.name == name:
            return adapter
    if _FALLBACK_MODEL_ADAPTER.name == name:
        return _FALLBACK_MODEL_ADAPTER
    available = ", ".join(adapter.name for adapter in _MODEL_ADAPTERS) or "<empty>"
    raise KeyError("Unknown model adapter '{}'. Available: {}.".format(name, available))


def registered_model_adapters() -> tuple[BaseModelAdapter, ...]:
    return tuple(_MODEL_ADAPTERS) + (_FALLBACK_MODEL_ADAPTER,)


def resolve_model_adapter(
    model: PreTrainedModel,
    adapter: str | BaseModelAdapter | None = None,
) -> BaseModelAdapter:
    if adapter is None:
        for candidate in _MODEL_ADAPTERS:
            if candidate.matches(model):
                return candidate
        if _FALLBACK_MODEL_ADAPTER.matches(model):
            return _FALLBACK_MODEL_ADAPTER
        available = ", ".join(candidate.name for candidate in registered_model_adapters()) or "<empty>"
        raise TypeError(
            "No registered model adapter matches model type {}. Registered adapters: {}.".format(
                type(model).__name__,
                available,
            )
        )

    resolved = get_model_adapter(adapter) if isinstance(adapter, str) else adapter
    resolved.ensure_supported(model)
    return resolved


for adapter_cls in (
    LlamaModelAdapter,
    Qwen2ModelAdapter,
    _load_optional_adapter("qwen3", "Qwen3ModelAdapter"),
    MistralModelAdapter,
):
    if adapter_cls is not None:
        register_model_adapter(adapter_cls())
