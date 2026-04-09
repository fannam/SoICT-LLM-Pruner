from __future__ import annotations

import pytest
from tests.fixtures.synthetic_models import SyntheticCausalLM, SyntheticConfig


def test_public_namespaces_smoke_import():
    from carve_lm import __version__
    from carve_lm.adapters import BaseModelAdapter
    from carve_lm.core import ESTIMATOR_REGISTRY
    from carve_lm.estimators import ActivationEstimator, create_estimator
    from carve_lm.evaluation import LLMMeasurer
    from carve_lm.pruners import WidthGroupConfig, WidthPruner
    from carve_lm.pruners.structured import StructuredBlockPruner

    assert __version__
    assert BaseModelAdapter is not None
    assert ESTIMATOR_REGISTRY is not None
    assert create_estimator is not None
    assert ActivationEstimator is not None
    assert WidthPruner is not None
    assert WidthGroupConfig is not None
    assert StructuredBlockPruner is not None
    assert LLMMeasurer is not None


def test_canonical_registry_names_are_listed_by_default():
    from carve_lm.estimators import available_estimators
    from carve_lm.pruners import available_pruners

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


def test_factories_resolve_legacy_aliases_with_warnings():
    from carve_lm.estimators import create_estimator
    from carve_lm.pruners import create_pruner

    model = SyntheticCausalLM(SyntheticConfig())

    with pytest.deprecated_call(match="element.weight_magnitude"):
        estimator = create_estimator("element.weight_magnitude", model, device="cpu")
    with pytest.deprecated_call(match="element"):
        pruner = create_pruner("element", model, device="cpu")

    assert estimator is not None
    assert pruner is not None


def test_distillation_namespace_smoke_import():
    from carve_lm.distillation import HybridDistiller, HybridOTDistiller, LogitsDistiller, OTConfig

    assert HybridDistiller is not None
    assert HybridOTDistiller is not None
    assert LogitsDistiller is not None
    assert OTConfig is not None


def test_teacher_correction_namespace_smoke_import():
    pytest.importorskip("accelerate")

    from carve_lm.distillation import TeacherCorrection

    assert TeacherCorrection is not None
