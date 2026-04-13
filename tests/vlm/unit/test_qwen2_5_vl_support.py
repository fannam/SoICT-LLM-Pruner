from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tests.vlm.fixtures.synthetic_vlm import (
    SyntheticVLMModel,
    SyntheticVLMModelAdapter,
    ensure_registered_synthetic_vlm_adapter,
    make_synthetic_vlm,
)
from torch.utils.data import DataLoader

from carve_lm.vlm.language.adapters import registered_model_adapters
from carve_lm.vlm.language.estimators._base import _BaseBlockPerplexityEstimator
from carve_lm.vlm.language.pruners import (
    DepthLayerConfig,
    DepthLayerPruner,
    EstimatorSpec,
    WidthGroupConfig,
    WidthGroupPruner,
    WidthPruner,
)


def make_dataloader(*samples: dict) -> DataLoader:
    return DataLoader(list(samples), batch_size=1)


def clone_visual_state(model: SyntheticVLMModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.visual.state_dict().items()
    }


def assert_visual_state_equal(model: SyntheticVLMModel, expected: dict[str, torch.Tensor]) -> None:
    actual = model.visual.state_dict()
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value)


def test_qwen2_5_vl_adapter_registration_matches_transformers_support():
    transformers = pytest.importorskip("transformers")
    names = {adapter.name for adapter in registered_model_adapters()}
    has_qwen2_5_vl = hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")

    if has_qwen2_5_vl:
        assert "qwen2_5_vl" in names
    else:
        assert "qwen2_5_vl" not in names


def test_block_perplexity_preserves_multimodal_batch_keys():
    model = make_synthetic_vlm(num_hidden_layers=1)
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "pixel_values": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        }
    )

    estimator = _BaseBlockPerplexityEstimator(
        model,
        tokenizer=object(),
        block_size=1,
        device="cpu",
        model_adapter=SyntheticVLMModelAdapter(),
    )

    scores = estimator.estimate(dataloader, n_samples=1)

    assert len(scores) == 1


def test_width_pruner_keeps_visual_state_untouched():
    model = make_synthetic_vlm(num_hidden_layers=1)
    expected_visual_state = clone_visual_state(model)
    pruner = WidthPruner(
        model,
        device="cpu",
        model_adapter=SyntheticVLMModelAdapter(),
    )

    pruned_model = pruner.prune_attention_group(
        group_importance={0: torch.tensor([0.1, 1.0])},
        target_group=1,
    )

    assert pruned_model is not None
    assert pruned_model.config.num_attention_heads == 2
    assert pruned_model.config.text_config.num_attention_heads == 2
    assert pruned_model.model.layers[0].self_attn.k_proj.out_features == 2
    assert_visual_state_equal(pruned_model, expected_visual_state)


def test_structured_roundtrip_preserves_visual_state_and_uses_vlm_manifest(tmp_path: Path):
    ensure_registered_synthetic_vlm_adapter()
    model = make_synthetic_vlm(num_hidden_layers=1)
    expected_visual_state = clone_visual_state(model)

    pruner = WidthGroupPruner(
        model,
        WidthGroupConfig(
            pruning_ratio=0.5,
            estimator=EstimatorSpec("random.group", {"seed": 0}),
        ),
        device="cpu",
        model_adapter="synthetic_vlm",
    )

    context = pruner.discover()
    scores = pruner.estimate(dataloader=None)
    plan = pruner.select(scores)
    result = pruner.apply(plan)

    save_dir = tmp_path / "vlm"
    pruner.save_pruned(save_dir, result)
    loaded = WidthGroupPruner.load_pruned(save_dir, device="cpu")

    assert (save_dir / "vlm_pruner_manifest.json").exists()
    manifest = json.loads((save_dir / "vlm_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width.group"
    assert manifest["adapter_name"] == "synthetic_vlm"
    assert len(context.groups) >= 2
    assert_visual_state_equal(loaded.model, expected_visual_state)


def test_vlm_config_mutators_cover_group_channel_and_depth_pruning():
    adapter = SyntheticVLMModelAdapter()

    group_model = make_synthetic_vlm(num_hidden_layers=1)
    group_pruner = WidthPruner(group_model, device="cpu", model_adapter=adapter)
    group_pruned = group_pruner.prune_attention_group(
        group_importance={0: torch.tensor([0.1, 1.0])},
        target_group=1,
    )
    assert group_pruned.config.num_attention_heads == 2
    assert group_pruned.config.text_config.num_attention_heads == 2
    assert isinstance(SyntheticVLMModel(group_pruned.config), SyntheticVLMModel)

    channel_model = make_synthetic_vlm(num_hidden_layers=1, hidden_size=8)
    channel_pruner = WidthPruner(channel_model, device="cpu", model_adapter=adapter)
    channel_pruned = channel_pruner.prune_embeddings(
        torch.arange(channel_model.config.hidden_size, dtype=torch.float32),
        target_embedding_size=4,
    )
    assert channel_pruned.config.hidden_size == 4
    assert channel_pruned.config.text_config.hidden_size == 4
    assert isinstance(SyntheticVLMModel(channel_pruned.config), SyntheticVLMModel)

    depth_model = make_synthetic_vlm(num_hidden_layers=2)
    depth_pruner = DepthLayerPruner(
        depth_model,
        DepthLayerConfig(target_num_layers=1),
        device="cpu",
        model_adapter=adapter,
    )
    depth_result = depth_pruner.apply()
    assert depth_result.model.config.num_hidden_layers == 1
    assert depth_result.model.config.text_config.num_hidden_layers == 1
    rebuilt = SyntheticVLMModel(depth_result.model.config)
    assert len(rebuilt.model.layers) == 1
