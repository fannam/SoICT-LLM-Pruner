from __future__ import annotations


def test_public_namespaces_smoke_import():
    from soict_llm_pruner import __version__
    from soict_llm_pruner.adapters import BaseModelAdapter
    from soict_llm_pruner.core import ESTIMATOR_REGISTRY
    from soict_llm_pruner.estimators import create_estimator
    from soict_llm_pruner.evaluation import LLMMeasurer
    from soict_llm_pruner.pruners import ElementPruner
    from soict_llm_pruner.pruners.structured import StructuredBlockPruner

    assert __version__
    assert BaseModelAdapter is not None
    assert ESTIMATOR_REGISTRY is not None
    assert create_estimator is not None
    assert ElementPruner is not None
    assert StructuredBlockPruner is not None
    assert LLMMeasurer is not None


def test_distillation_namespace_smoke_import():
    import pytest

    pytest.importorskip("accelerate")
    pytest.importorskip("matplotlib")

    from soict_llm_pruner.distillation import HybridDistiller, LogitsDistiller, TeacherCorrection

    assert HybridDistiller is not None
    assert LogitsDistiller is not None
    assert TeacherCorrection is not None
