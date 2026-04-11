from __future__ import annotations

import importlib

import pytest
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig


def test_root_namespace_is_minimal_and_lazy():
    import carve_lm
    from carve_lm import __version__

    assert __version__
    assert carve_lm.llm is not None
    assert carve_lm.vlm is not None
    with pytest.raises(AttributeError):
        _ = carve_lm.estimators


def test_llm_public_namespaces_smoke_import():
    from carve_lm import __version__
    from carve_lm.llm.adapters import BaseModelAdapter
    from carve_lm.llm.core import ESTIMATOR_REGISTRY
    from carve_lm.llm.distillation import HybridDistiller, HybridOTDistiller, LogitsDistiller, OTConfig
    from carve_lm.llm.estimators import ActivationEstimator, create_estimator
    from carve_lm.llm.evaluation import LLMMeasurer
    from carve_lm.llm.pruners import WidthGroupConfig, WidthPruner
    from carve_lm.llm.pruners.structured import StructuredBlockPruner

    assert __version__
    assert BaseModelAdapter is not None
    assert ESTIMATOR_REGISTRY is not None
    assert create_estimator is not None
    assert ActivationEstimator is not None
    assert WidthPruner is not None
    assert WidthGroupConfig is not None
    assert StructuredBlockPruner is not None
    assert HybridDistiller is not None
    assert HybridOTDistiller is not None
    assert LogitsDistiller is not None
    assert OTConfig is not None
    assert LLMMeasurer is not None


def test_vlm_public_namespaces_smoke_import():
    from carve_lm.vlm.adapters import BaseModelAdapter, resolve_model_adapter
    from carve_lm.vlm.distillation import HybridDistiller, HybridOTDistiller, LogitsDistiller, OTConfig
    from carve_lm.vlm.estimators import ActivationEstimator
    from carve_lm.vlm.evaluation import VLMMeasurer
    from carve_lm.vlm.pruners import WidthGroupConfig, WidthPruner

    assert BaseModelAdapter is not None
    assert resolve_model_adapter is not None
    assert ActivationEstimator is not None
    assert WidthPruner is not None
    assert WidthGroupConfig is not None
    assert HybridDistiller is not None
    assert HybridOTDistiller is not None
    assert LogitsDistiller is not None
    assert OTConfig is not None
    assert VLMMeasurer is not None


def test_old_root_pruning_imports_are_removed():
    for module_name in (
        "carve_lm.adapters",
        "carve_lm.core",
        "carve_lm.distillation",
        "carve_lm.estimators",
        "carve_lm.evaluation",
        "carve_lm.pruners",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_llm_canonical_registry_names_are_listed_by_default():
    from carve_lm.llm.estimators import available_estimators
    from carve_lm.llm.pruners import available_pruners

    assert available_estimators() == (
        "activation.element",
        "magnitude.channel",
        "magnitude.element",
        "magnitude.group",
        "perplexity.block",
        "random.group",
        "similarity.block",
        "similarity.layer",
        "taylor.group",
    )
    assert "element.activation" not in available_estimators()
    assert "element" not in available_pruners()
    assert available_pruners() == (
        "component",
        "depth.block",
        "depth.layer",
        "width",
        "width.channel",
        "width.group",
    )


def test_llm_factories_resolve_legacy_aliases_with_warnings():
    from carve_lm.llm.estimators import create_estimator
    from carve_lm.llm.pruners import create_pruner

    model = SyntheticCausalLM(SyntheticConfig())

    with pytest.deprecated_call(match="element.weight_magnitude"):
        estimator = create_estimator("element.weight_magnitude", model, device="cpu")
    with pytest.deprecated_call(match="element"):
        pruner = create_pruner("element", model, device="cpu")

    assert estimator is not None
    assert pruner is not None

def test_llm_teacher_correction_namespace_smoke_import():
    pytest.importorskip("accelerate")

    from carve_lm.llm.distillation import TeacherCorrection

    assert TeacherCorrection is not None


def test_vlm_teacher_correction_namespace_smoke_import():
    pytest.importorskip("accelerate")

    from carve_lm.vlm.distillation import TeacherCorrection

    assert TeacherCorrection is not None
