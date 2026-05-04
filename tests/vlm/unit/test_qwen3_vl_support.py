from __future__ import annotations

import json
from pathlib import Path

import torch
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_qwen3_vl
from torch.utils.data import DataLoader

from carve_lm.vlm.components.language.adapters import get_model_adapter as get_language_adapter
from carve_lm.vlm.components.language.adapters import registered_model_adapters
from carve_lm.vlm.components.language.estimators import create_estimator as create_language_estimator
from carve_lm.vlm.components.language.pruners import (
    EstimatorSpec,
    WidthChannelConfig,
    WidthChannelPruner,
)
from carve_lm.vlm.components.merger.adapters import get_model_adapter as get_merger_adapter
from carve_lm.vlm.components.merger.estimators import create_estimator as create_merger_estimator
from carve_lm.vlm.components.merger.pruners import BridgeChannelConfig, BridgeChannelPruner
from carve_lm.vlm.components.merger.pruners import WidthConfig as MergerWidthConfig
from carve_lm.vlm.components.merger.pruners import WidthPruner as MergerWidthPruner
from carve_lm.vlm.components.vision.adapters import get_model_adapter as get_vision_adapter
from carve_lm.vlm.components.vision.estimators import create_estimator as create_vision_estimator
from carve_lm.vlm.components.vision.pruners import WidthPruner as VisionWidthPruner
from carve_lm.vlm.distillation import HybridDistiller


def make_dataloader(*samples: dict) -> DataLoader:
    return DataLoader(list(samples), batch_size=1)


def make_qwen3_batch(offset: int = 0, *, batch_size: int = 1) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [[(offset + 1) % 16, (offset + 2) % 16, (offset + 3) % 16, (offset + 4) % 16]]
        * batch_size,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.arange(batch_size * 32, dtype=torch.float32).reshape(batch_size, 4, 8)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
        "pixel_values": pixel_values,
        "token_type_ids": torch.zeros_like(input_ids),
    }


def assert_qwen3_vl_forward_runs(model) -> None:
    outputs = model(**{key: value for key, value in make_qwen3_batch().items() if key != "token_type_ids"})
    assert outputs["logits"].shape == (1, 4, model.config.text_config.vocab_size)


def make_qwen3_forward_inputs(offset: int = 0, *, batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in make_qwen3_batch(offset, batch_size=batch_size).items()
        if key != "token_type_ids"
    }


def test_qwen3_vl_adapters_are_registered_and_resolve_synthetic_layout():
    names = {adapter.name for adapter in registered_model_adapters()}
    model = make_synthetic_qwen3_vl(num_hidden_layers=1)

    assert "qwen3_vl" in names
    assert get_language_adapter("qwen3_vl").matches(model)
    assert get_vision_adapter("qwen3_vl").matches(model)
    assert get_merger_adapter("qwen3_vl").matches(model)


def test_qwen3_vl_language_channel_pruning_keeps_attention_head_dim_norms():
    model = make_synthetic_qwen3_vl(num_hidden_layers=1)
    adapter = get_language_adapter("qwen3_vl")
    estimator = create_language_estimator(
        "magnitude.element",
        model,
        device="cpu",
        model_adapter=adapter,
    )
    scores = estimator.estimate_embedding_channels(agg="l1")
    assert scores["embedding_channels"].numel() == model.config.text_config.hidden_size

    pruner = WidthChannelPruner(
        model,
        WidthChannelConfig(
            pruning_ratio=0.5,
            estimator=EstimatorSpec("magnitude.element", {"agg": "l1"}),
        ),
        device="cpu",
        model_adapter="qwen3_vl",
    )
    context = pruner.discover()
    selected_scores = {
        group.group_id: (10.0 if group.local_idx in {1, 3, 5, 7} else 0.0)
        for group in context.groups
    }
    result = pruner.apply(pruner.select(selected_scores))
    pruned = result.model
    attention = pruned.model.language_model.layers[0].self_attn

    assert context.metadata["channel_group_kind"] == "hidden_stream"
    assert pruned.config.text_config.hidden_size == 4
    assert attention.head_dim == 2
    assert attention.q_norm.weight.numel() == 2
    assert attention.k_norm.weight.numel() == 2
    assert attention.q_proj.in_features == 4
    assert attention.q_proj.out_features == 8
    assert attention.k_proj.in_features == 4
    assert attention.k_proj.out_features == 4
    assert attention.o_proj.in_features == 8
    assert attention.o_proj.out_features == 4
    assert result.plan.metadata["selected_residual_indices"] == [1, 3, 5, 7]


def test_qwen3_vl_vision_estimators_and_pruners_handle_two_linear_mlp():
    model = make_synthetic_qwen3_vl(vision_depth=1)
    dataloader = make_dataloader(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "pixel_values": torch.arange(32, dtype=torch.float32).reshape(4, 8),
        }
    )
    estimator = create_vision_estimator(
        "magnitude.element",
        model,
        device="cpu",
        model_adapter="qwen3_vl",
    )

    attention_scores = estimator.estimate_attention_heads(agg="l1")
    mlp_scores = estimator.estimate_mlp_neurons(agg="l1")
    hidden_scores = estimator.estimate_hidden_channels(agg="l1")

    assert attention_scores[0].numel() == model.config.vision_config.num_heads
    assert mlp_scores[0].numel() == model.config.vision_config.intermediate_size
    assert hidden_scores["vision_block0_hidden_channels"].numel() == model.config.vision_config.hidden_size

    activation = create_vision_estimator(
        "activation.element",
        model,
        device="cpu",
        model_adapter="qwen3_vl",
    )
    assert activation.estimate_mlp_neurons(dataloader, agg="sum")[0].numel() == 6

    pruner = VisionWidthPruner(model, device="cpu", model_adapter="qwen3_vl")
    mlp_pruned = pruner.prune_mlp_neurons(
        {0: torch.tensor([0.1, 3.0, 0.2, 2.0, 0.3, 1.0])},
        target_num_neurons=3,
    )
    mlp = mlp_pruned.model.visual.blocks[0].mlp
    assert mlp.linear_fc1.out_features == 3
    assert mlp.linear_fc2.in_features == 3
    assert_qwen3_vl_forward_runs(mlp_pruned)

    hidden_pruned = pruner.prune_hidden_channels(
        {"vision": torch.tensor([0.1, 5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0])},
        target_hidden_size=4,
    )
    primary_merger = hidden_pruned.model.visual.merger
    deepstack_merger = hidden_pruned.model.visual.deepstack_merger_list[0]
    assert hidden_pruned.config.vision_config.hidden_size == 4
    assert hidden_pruned.model.visual.blocks[0].mlp.linear_fc2.out_features == 4
    assert primary_merger.norm.weight.numel() == 4
    assert primary_merger.linear_fc1.in_features == 16
    assert primary_merger.hidden_size == 16
    assert deepstack_merger.norm.weight.numel() == 16
    assert deepstack_merger.linear_fc1.in_features == 16
    assert deepstack_merger.hidden_size == 16
    assert_qwen3_vl_forward_runs(hidden_pruned)


def test_qwen3_vl_merger_estimators_and_pruners_aggregate_deepstack_mergers():
    model = make_synthetic_qwen3_vl()
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
        model_adapter="qwen3_vl",
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
        model_adapter="qwen3_vl",
    )
    assert activation.estimate_output_channels(dataloader, agg="sum")["merger_output_channels"].numel() == 8

    pruner = MergerWidthPruner(model, device="cpu", model_adapter="qwen3_vl")
    pruned = pruner.prune_intermediate_channels(
        {"merger_intermediate_channels": torch.arange(32, dtype=torch.float32)},
        target_num_channels=8,
    )

    assert pruned.model.visual.merger.linear_fc1.out_features == 8
    assert pruned.model.visual.merger.linear_fc2.in_features == 8
    assert pruned.model.visual.deepstack_merger_list[0].linear_fc1.out_features == 8
    assert pruned.model.visual.deepstack_merger_list[0].linear_fc2.in_features == 8
    assert_qwen3_vl_forward_runs(pruned)


def test_qwen3_vl_bridge_pruning_slices_primary_and_deepstack_merger_outputs():
    model = make_synthetic_qwen3_vl(num_hidden_layers=1)
    pruner = BridgeChannelPruner(
        model,
        BridgeChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter="qwen3_vl",
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
    assert pruned.model.visual.merger.linear_fc2.out_features == 4
    assert pruned.model.visual.deepstack_merger_list[0].linear_fc2.out_features == 4
    assert result.plan.metadata["selected_residual_indices"] == [1, 3, 5, 7]
    assert_qwen3_vl_forward_runs(pruned)


def test_qwen3_vl_merger_pruner_manifest_roundtrip(tmp_path: Path):
    model = make_synthetic_qwen3_vl(vision_depth=1)
    pruner = VisionWidthPruner(model, device="cpu", model_adapter="qwen3_vl")
    pruned = pruner.prune_mlp_neurons(
        {0: torch.tensor([0.1, 3.0, 0.2, 2.0, 0.3, 1.0])},
        target_num_neurons=3,
    )
    assert_qwen3_vl_forward_runs(pruned)

    structured = MergerWidthPruner(
        pruned,
        MergerWidthConfig(pruning_ratio=0.75),
        device="cpu",
        model_adapter="qwen3_vl",
    )
    context = structured.discover()
    scores = {group.group_id: float(group.local_idx) for group in context.groups}
    result = structured.apply(structured.select(scores))
    expected_logits = result.model(**make_qwen3_forward_inputs())["logits"]

    save_dir = tmp_path / "qwen3_merger"
    structured.save_pruned(save_dir, result)
    loaded = MergerWidthPruner.load_pruned(save_dir, device="cpu")
    actual_logits = loaded.model(**make_qwen3_forward_inputs())["logits"]

    torch.testing.assert_close(actual_logits, expected_logits)
    manifest = json.loads((save_dir / "vlm_merger_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width"
    assert manifest["adapter_name"] == "qwen3_vl"


def test_qwen3_vl_hybrid_distillation_uses_nested_text_config_and_filters_token_type_ids():
    torch.manual_seed(0)
    teacher = make_synthetic_qwen3_vl(hidden_size=8, num_hidden_layers=2)
    student = make_synthetic_qwen3_vl(hidden_size=4, num_hidden_layers=1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)

    distiller = HybridDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
    )

    val_history = distiller.distill(
        train_loader=[make_qwen3_batch(0, batch_size=2)],
        val_loader=[make_qwen3_batch(7, batch_size=2)],
        device_teacher="cpu",
        device_student="cpu",
        epochs=1,
        grad_accumulation_steps=1,
        block_layers_to_prune=[1],
    )

    assert distiller.teacher_kept_layers == [0]
    assert len(val_history) == 1
    assert distiller.history["train_loss"]
