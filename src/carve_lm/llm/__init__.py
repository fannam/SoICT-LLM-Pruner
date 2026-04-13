from __future__ import annotations

from importlib import import_module

__all__ = [
    "adapters",
    "auto_model",
    "core",
    "distillation",
    "estimators",
    "evaluation",
    "pruners",
]


def __getattr__(name: str):
    if name in __all__:
        return import_module(".{}".format(name), __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
