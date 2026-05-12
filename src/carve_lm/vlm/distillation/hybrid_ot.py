from __future__ import annotations

from carve_lm._distillation.hybrid_ot import HybridOTDistillerMixin

from .hybrid import HybridDistiller


class HybridOTDistiller(HybridOTDistillerMixin, HybridDistiller):
    pass


__all__ = ["HybridOTDistiller"]
