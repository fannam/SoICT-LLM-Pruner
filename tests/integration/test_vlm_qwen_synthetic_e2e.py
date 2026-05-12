from __future__ import annotations

import pytest
import torch

from carve_lm.vlm.components.merger.pruners import BridgeChannelConfig, BridgeChannelPruner
from carve_lm.vlm.components.merger.pruners import WidthPruner as MergerWidthPruner
from carve_lm.vlm.components.vision.pruners import WidthPruner as VisionWidthPruner
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_qwen2_5_vl, make_synthetic_qwen3_vl


def make_forward_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    }


def assert_forward_runs(model) -> None:
    outputs = model(**make_forward_inputs())
    assert outputs["logits"].shape == (1, 4, model.config.text_config.vocab_size)


@pytest.mark.parametrize(
    ("model_name", "factory"),
    [
        ("qwen2_5_vl", make_synthetic_qwen2_5_vl),
        ("qwen3_vl", make_synthetic_qwen3_vl),
    ],
)
def test_synthetic_qwen_vlm_bridge_vision_and_merger_pruning_run_end_to_end(model_name, factory):
    bridge_model = factory(num_hidden_layers=1, vision_depth=1)
    bridge_pruner = BridgeChannelPruner(
        bridge_model,
        BridgeChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter=model_name,
    )
    bridge_context = bridge_pruner.discover()
    bridge_scores = {
        group.group_id: (10.0 if group.local_idx in {1, 3, 5, 7} else 0.0)
        for group in bridge_context.groups
    }
    bridge_result = bridge_pruner.apply(bridge_pruner.select(bridge_scores))

    assert bridge_result.model.config.text_config.hidden_size == 4
    assert bridge_result.plan.metadata["selected_residual_indices"] == [1, 3, 5, 7]
    assert_forward_runs(bridge_result.model)

    component_model = factory(num_hidden_layers=1, vision_depth=1)
    vision_pruner = VisionWidthPruner(component_model, device="cpu", model_adapter=model_name)
    vision_pruned = vision_pruner.prune_mlp_neurons(
        {0: torch.tensor([0.1, 3.0, 0.2, 2.0, 0.3, 1.0])},
        target_num_neurons=3,
    )
    assert vision_pruned.config.vision_config.intermediate_size == 3
    assert_forward_runs(vision_pruned)

    merger_pruner = MergerWidthPruner(vision_pruned, device="cpu", model_adapter=model_name)
    merger_pruned = merger_pruner.prune_intermediate_channels(
        {"merger_intermediate_channels": torch.arange(32, dtype=torch.float32)},
        target_num_channels=8,
    )
    assert_forward_runs(merger_pruned)
