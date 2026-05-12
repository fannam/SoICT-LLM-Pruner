from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from carve_lm.vlm.components.merger.pruners import BridgeChannelConfig, BridgeChannelPruner
from carve_lm.vlm.components.merger.pruners import WidthConfig as MergerWidthConfig
from carve_lm.vlm.components.merger.pruners import WidthPruner as MergerWidthPruner
from carve_lm.vlm.components.vision.pruners import WidthChannelConfig as VisionWidthChannelConfig
from carve_lm.vlm.components.vision.pruners import WidthChannelPruner as VisionWidthChannelPruner
from tests.vlm.fixtures.synthetic_vlm import make_synthetic_qwen2_5_vl, make_synthetic_qwen3_vl


def make_forward_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.arange(32, dtype=torch.float32).reshape(1, 4, 8),
    }


def assert_roundtrip_logits(result, loaded) -> None:
    expected = result.model(**make_forward_inputs())["logits"]
    actual = loaded.model(**make_forward_inputs())["logits"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("model_name", "factory"),
    [
        ("qwen2_5_vl", make_synthetic_qwen2_5_vl),
        ("qwen3_vl", make_synthetic_qwen3_vl),
    ],
)
def test_synthetic_qwen_bridge_pruner_persistence_roundtrip(model_name, factory, tmp_path: Path):
    model = factory(num_hidden_layers=1, vision_depth=1)
    pruner = BridgeChannelPruner(
        model,
        BridgeChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter=model_name,
    )
    context = pruner.discover()
    scores = {
        group.group_id: (10.0 if group.local_idx in {1, 3, 5, 7} else 0.0)
        for group in context.groups
    }
    result = pruner.apply(pruner.select(scores))

    save_dir = tmp_path / f"{model_name}_bridge"
    pruner.save_pruned(save_dir, result)
    loaded = BridgeChannelPruner.load_pruned(save_dir, device="cpu")

    assert_roundtrip_logits(result, loaded)
    manifest = json.loads((save_dir / "vlm_merger_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width.bridge"
    assert manifest["adapter_name"] == model_name


@pytest.mark.parametrize(
    ("model_name", "factory"),
    [
        ("qwen2_5_vl", make_synthetic_qwen2_5_vl),
        ("qwen3_vl", make_synthetic_qwen3_vl),
    ],
)
def test_synthetic_qwen_vision_pruner_persistence_roundtrip(model_name, factory, tmp_path: Path):
    model = factory(num_hidden_layers=1, vision_depth=1)
    pruner = VisionWidthChannelPruner(
        model,
        VisionWidthChannelConfig(pruning_ratio=0.5),
        device="cpu",
        model_adapter=model_name,
    )
    context = pruner.discover()
    scores = {group.group_id: float(group.local_idx) for group in context.groups}
    result = pruner.apply(pruner.select(scores))

    save_dir = tmp_path / f"{model_name}_vision"
    pruner.save_pruned(save_dir, result)
    loaded = VisionWidthChannelPruner.load_pruned(save_dir, device="cpu")

    assert_roundtrip_logits(result, loaded)
    manifest = json.loads((save_dir / "vlm_vision_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width.channel"
    assert manifest["adapter_name"] == model_name


@pytest.mark.parametrize(
    ("model_name", "factory"),
    [
        ("qwen2_5_vl", make_synthetic_qwen2_5_vl),
        ("qwen3_vl", make_synthetic_qwen3_vl),
    ],
)
def test_synthetic_qwen_merger_pruner_persistence_roundtrip(model_name, factory, tmp_path: Path):
    model = factory(num_hidden_layers=1, vision_depth=1)
    pruner = MergerWidthPruner(
        model,
        MergerWidthConfig(pruning_ratio=0.75),
        device="cpu",
        model_adapter=model_name,
    )
    context = pruner.discover()
    scores = {group.group_id: float(group.local_idx) for group in context.groups}
    result = pruner.apply(pruner.select(scores))

    save_dir = tmp_path / f"{model_name}_merger"
    pruner.save_pruned(save_dir, result)
    loaded = MergerWidthPruner.load_pruned(save_dir, device="cpu")

    assert_roundtrip_logits(result, loaded)
    manifest = json.loads((save_dir / "vlm_merger_pruner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_pruner"] == "width"
    assert manifest["adapter_name"] == model_name
