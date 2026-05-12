from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "examples.pruning.basic_usage",
        "examples.structured.llm_pruner_usage",
        "examples.evaluation.measure_latency",
        "scripts.recovery.finetune_llama",
        "scripts.recovery.finetune_qwen",
        "scripts.recovery.teacher_correction_accelerate",
        "scripts.validation.validate_real_qwen_vlm",
    ],
)
def test_examples_and_recovery_scripts_import_without_side_effects(module_name):
    module = importlib.import_module(module_name)
    assert module is not None
