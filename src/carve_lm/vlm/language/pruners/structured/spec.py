from __future__ import annotations

from ...adapters import BaseModelAdapter, import_object, object_path, resolve_model_adapter

ArchitectureSpec = BaseModelAdapter


def resolve_architecture_spec(model):
    return resolve_model_adapter(model)


__all__ = [
    "ArchitectureSpec",
    "import_object",
    "object_path",
    "resolve_architecture_spec",
]
