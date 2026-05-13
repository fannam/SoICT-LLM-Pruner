from __future__ import annotations

import json
from pathlib import Path

import torch
from tests.vlm.fixtures.synthetic_vlm import (
    SyntheticVLMModel,
    SyntheticVLMModelAdapter,
    ensure_registered_synthetic_vlm_adapter,
    make_synthetic_qwen2_5_vl,
    make_synthetic_vlm,
)
from torch.utils.data import DataLoader

from carve_lm.vlm.components.language.adapters import get_model_adapter, registered_model_adapters
from carve_lm.vlm.components.language.estimators import create_estimator as create_language_estimator
from carve_lm.vlm.components.language.estimators._base import _BaseBlockPerplexityEstimator
from carve_lm.vlm.components.language.pruners import (
    DepthLayerConfig,
    DepthLayerPruner,
    EstimatorSpec,
    WidthGroupConfig,
    WidthGroupPruner,
    WidthPruner,
)
from carve_lm.vlm.components.merger.estimators import create_estimator as create_merger_estimator
from carve_lm.vlm.components.merger.pruners import (
    BridgeChannelConfig,
    BridgeChannelPruner,
)
from carve_lm.vlm.components.merger.pruners import (
    WidthConfig as MergerWidthConfig,
)
from carve_lm.vlm.components.merger.pruners import (
    WidthPruner as MergerWidthPruner,
)
from carve_lm.vlm.components.vision.estimators import create_estimator as create_vision_estimator
from carve_lm.vlm.components.vision.pruners import (
    DepthLayerConfig as VisionDepthLayerConfig,
)
from carve_lm.vlm.components.vision.pruners import (
    DepthLayerPruner as VisionDepthLayerPruner,
)
from carve_lm.vlm.components.vision.pruners import (
    WidthChannelConfig as VisionWidthChannelConfig,
)
from carve_lm.vlm.components.vision.pruners import (
    WidthChannelPruner as VisionWidthChannelPruner,
)
from carve_lm.vlm.components.vision.pruners import (
    WidthPruner as VisionWidthPruner,
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


def assert_qwen2_5_vl_forward_runs(model) -> None:
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        pixel_values=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    )
    assert outputs["logits"].shape == (1, 4, model.config.text_config.vocab_size)


def test_qwen2_5_vl_adapter_is_registered():
    names = {adapter.name for adapter in registered_model_adapters()}

    assert "qwen2_5_vl" in names


def test_qwen2_5_vl_language_estimator_uses_nested_language_model():
    model = make_synthetic_qwen2_5_vl(num_hidden_layers=1)
    adapter = get_model_adapter("qwen2_5_vl")

    assert adapter.get_layers(model) is model.model.language_model.layers

    estimator = create_language_estimator(
        "magnitude.element",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    scores = estimator.estimate_attention_groups(agg="l1")

    assert set(scores) == {0}
    assert scores[0].numel() == model.config.text_config.num_key_value_heads


def test_qwen2_5_vl_vision_estimators_score_fused_qkv_blocks():
    model = make_synthetic_qwen2_5_vl(vision_depth=1)
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "pixel_values": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }
    )

    magnitude = create_vision_estimator(
        "magnitude.element",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    attention_scores = magnitude.estimate_attention_heads(agg="l1")
    mlp_scores = magnitude.estimate_mlp_neurons(agg="l1")
    hidden_scores = magnitude.estimate_hidden_channels(agg="l1")

    assert attention_scores[0].numel() == model.config.vision_config.num_heads
    assert mlp_scores[0].numel() == model.config.vision_config.intermediate_size
    assert hidden_scores["vision_block0_hidden_channels"].numel() == model.config.vision_config.hidden_size

    activation = create_vision_estimator(
        "activation.element",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    activation_scores = activation.estimate_attention_heads(dataloader, agg="sum")

    assert activation_scores[0].numel() == model.config.vision_config.num_heads


def test_qwen2_5_vl_vision_similarity_estimators_score_blocks():
    model = make_synthetic_qwen2_5_vl(vision_depth=2)
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "pixel_values": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }
    )

    layer_estimator = create_vision_estimator(
        "similarity.layer",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    layer_scores = layer_estimator.estimate(dataloader)

    assert len(layer_scores["attention"]) == 2
    assert len(layer_scores["mlp"]) == 2

    block_estimator = create_vision_estimator(
        "similarity.block",
        model,
        block_size=1,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    block_scores = block_estimator.estimate(dataloader)

    assert len(block_scores) == 2


def test_qwen2_5_vl_merger_estimators_score_patch_merger_channels():
    model = make_synthetic_qwen2_5_vl()
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "pixel_values": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }
    )

    magnitude = create_merger_estimator(
        "magnitude.element",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    input_scores = magnitude.estimate_input_channels(agg="l1")
    intermediate_scores = magnitude.estimate_intermediate_channels(agg="l1")
    output_scores = magnitude.estimate_output_channels(agg="l1")

    assert input_scores["merger_input_channels"].numel() == model.config.vision_config.hidden_size
    assert intermediate_scores["merger_intermediate_channels"].numel() == 32
    assert output_scores["merger_output_channels"].numel() == model.config.text_config.hidden_size

    activation = create_merger_estimator(
        "activation.element",
        model,
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    activation_scores = activation.estimate_output_channels(dataloader, agg="sum")

    assert activation_scores["merger_output_channels"].numel() == model.config.text_config.hidden_size


def test_qwen2_5_vl_vision_attention_head_pruning_slices_fused_qkv():
    model = make_synthetic_qwen2_5_vl(vision_depth=1)
    pruner = VisionWidthPruner(model, device="cpu", model_adapter="qwen2_5_vl")

    pruned = pruner.prune_attention_heads(
        {0: torch.tensor([0.1, 10.0, 0.2, 9.0])},
        target_num_heads=2,
    )
    attention = pruned.model.visual.blocks[0].attn

    assert attention.num_heads == 2
    assert pruned.config.vision_config.num_heads == 2
    assert attention.qkv.out_features == 12
    assert attention.proj.in_features == 4
    assert_qwen2_5_vl_forward_runs(pruned)


def test_qwen2_5_vl_vision_mlp_pruning_slices_neurons():
    model = make_synthetic_qwen2_5_vl(vision_depth=1, vision_intermediate_size=6)
    pruner = VisionWidthPruner(model, device="cpu", model_adapter="qwen2_5_vl")

    pruned = pruner.prune_mlp_neurons(
        {0: torch.tensor([0.1, 3.0, 0.2, 2.0, 0.3, 1.0])},
        target_num_neurons=3,
    )
    mlp = pruned.model.visual.blocks[0].mlp

    assert pruned.config.vision_config.intermediate_size == 3
    assert mlp.gate_proj.out_features == 3
    assert mlp.up_proj.out_features == 3
    assert mlp.down_proj.in_features == 3
    assert_qwen2_5_vl_forward_runs(pruned)


def test_qwen2_5_vl_vision_depth_pruning_keeps_prefix_blocks():
    model = make_synthetic_qwen2_5_vl(vision_depth=3)
    pruner = VisionDepthLayerPruner(
        model,
        VisionDepthLayerConfig(target_num_layers=1),
        device="cpu",
        model_adapter="qwen2_5_vl",
    )

    result = pruner.run()

    assert len(result.model.model.visual.blocks) == 1
    assert result.model.config.vision_config.depth == 1
    assert_qwen2_5_vl_forward_runs(result.model)


def test_qwen2_5_vl_vision_hidden_channel_pruning_updates_merger_input_boundary():
    model = make_synthetic_qwen2_5_vl(vision_depth=1)
    pruner = VisionWidthPruner(model, device="cpu", model_adapter="qwen2_5_vl")
    scores = {"vision": torch.tensor([0.1, 5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0])}

    pruned = pruner.prune_hidden_channels(scores, target_hidden_size=4)
    block = pruned.model.visual.blocks[0]
    merger = pruned.model.visual.merger

    assert pruned.config.vision_config.hidden_size == 4
    assert pruned.model.visual.patch_embed.proj.out_features == 4
    assert block.norm1.weight.numel() == 4
    assert block.attn.qkv.in_features == 4
    assert block.attn.qkv.out_features == 12
    assert block.attn.proj.in_features == 4
    assert block.attn.proj.out_features == 4
    assert block.mlp.down_proj.out_features == 4
    assert merger.ln_q.weight.numel() == 4
    assert merger.mlp[0].in_features == 16
    assert merger.mlp[2].out_features == model.config.text_config.hidden_size
    assert_qwen2_5_vl_forward_runs(pruned)


def test_qwen2_5_vl_merger_intermediate_pruning_slices_mlp_boundary():
    model = make_synthetic_qwen2_5_vl()
    pruner = MergerWidthPruner(model, device="cpu", model_adapter="qwen2_5_vl")

    pruned = pruner.prune_intermediate_channels(
        {"merger_intermediate_channels": torch.arange(32, dtype=torch.float32)},
        target_num_channels=8,
    )
    merger = pruned.model.visual.merger

    assert merger.mlp[0].out_features == 8
    assert merger.mlp[2].in_features == 8
    assert_qwen2_5_vl_forward_runs(pruned)


def test_qwen2_5_vl_bridge_pruning_aligns_merger_output_and_language_hidden_channels():
    model = make_synthetic_qwen2_5_vl(num_hidden_layers=1)
    pruner = BridgeChannelPruner(
        model,
        BridgeChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    context = pruner.discover()
    scores = {
        group.group_id: (10.0 if group.local_idx in {1, 3, 5, 7} else 0.0)
        for group in context.groups
    }

    result = pruner.apply(pruner.select(scores))
    pruned = result.model

    assert pruned.config.text_config.hidden_size == 4
    assert pruned.model.language_model.embed_tokens.embedding_dim == 4
    assert pruned.model.visual.merger.mlp[2].out_features == 4
    assert result.plan.metadata["selected_residual_indices"] == [1, 3, 5, 7]
    assert_qwen2_5_vl_forward_runs(pruned)


def test_qwen2_5_vl_vision_pruner_manifest_roundtrip(tmp_path: Path):
    model = make_synthetic_qwen2_5_vl(vision_depth=1)
    pruner = VisionWidthChannelPruner(
        model,
        VisionWidthChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    context = pruner.discover()
    scores = {
        group.group_id: (10.0 if group.group_id == "vision.channel.bundle1" else 0.0)
        for group in context.groups
    }
    result = pruner.apply(pruner.select(scores))
    expected_logits = result.model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        pixel_values=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    )["logits"]

    save_dir = tmp_path / "vision"
    pruner.save_pruned(save_dir, result)
    loaded = VisionWidthChannelPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        pixel_values=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    )["logits"]

    torch.testing.assert_close(actual_logits, expected_logits)
    manifest = json.loads((save_dir / "vlm_vision_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width.channel"
    assert manifest["adapter_name"] == "qwen2_5_vl"


def test_qwen2_5_vl_merger_pruner_manifest_roundtrip(tmp_path: Path):
    model = make_synthetic_qwen2_5_vl()
    pruner = MergerWidthPruner(
        model,
        MergerWidthConfig(pruning_ratio=0.75),
        device="cpu",
        model_adapter="qwen2_5_vl",
    )
    context = pruner.discover()
    scores = {
        group.group_id: float(group.local_idx)
        for group in context.groups
    }
    result = pruner.apply(pruner.select(scores))
    expected_logits = result.model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        pixel_values=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    )["logits"]

    save_dir = tmp_path / "merger"
    pruner.save_pruned(save_dir, result)
    loaded = MergerWidthPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        pixel_values=torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    )["logits"]

    torch.testing.assert_close(actual_logits, expected_logits)
    manifest = json.loads((save_dir / "vlm_merger_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width"
    assert manifest["adapter_name"] == "qwen2_5_vl"


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
