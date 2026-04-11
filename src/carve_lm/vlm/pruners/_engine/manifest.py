from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...adapters import import_object, object_path
from .types import DiscoveryContext, PruningPlan, PruningResult
from .utils import json_dump, json_load

MANIFEST_FILENAME = "vlm_pruner_manifest.json"
STATE_FILENAME = "model_state.pt"


def build_manifest(
    *,
    mode: str,
    canonical_pruner: str,
    adapter_name: str,
    config,
    context: DiscoveryContext,
    plan: PruningPlan,
    pruned_model,
) -> dict[str, Any]:
    return {
        "version": 2,
        "pruning_mode": mode,
        "canonical_pruner": canonical_pruner,
        "adapter_name": adapter_name,
        "family_key": context.family_key,
        "model_class_path": context.model_class_path,
        "config_class_path": context.config_class_path,
        "base_config": context.base_config,
        "config_class": object_path(config.__class__),
        "config_payload": config.to_dict(),
        "discovery_context": context.to_dict(),
        "plan": plan.to_dict(),
        "pruning_history": list(getattr(pruned_model.config, "pruning_history", [])),
    }


def save_pruned_result(output_dir, result: PruningResult) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), output_path / STATE_FILENAME)
    json_dump(output_path / MANIFEST_FILENAME, result.manifest)
    return output_path


def load_pruned_result(pruner_cls, output_dir, device: str | None = None, dtype=None) -> PruningResult:
    output_path = Path(output_dir)
    manifest = json_load(output_path / MANIFEST_FILENAME)
    model_cls = import_object(manifest["model_class_path"])
    config_cls = import_object(manifest["config_class_path"])
    base_config = _config_from_dict(config_cls, manifest["base_config"])
    model = _instantiate_model(model_cls, base_config, device=device, dtype=dtype)

    if manifest.get("version", 1) >= 2:
        pruner_config_cls = import_object(manifest["config_class"])
        config = _config_from_dict(pruner_config_cls, manifest["config_payload"])
    else:
        config = pruner_cls.config_cls.from_dict(manifest["pruner_config"])

    pruner = pruner_cls(model=model, config=config, device=device or "cpu")
    context = DiscoveryContext.from_dict(manifest["discovery_context"])
    plan = PruningPlan.from_dict(manifest["plan"])
    pruner._last_context = context
    pruner._last_plan = plan
    result = pruner.apply(plan)
    state = torch.load(output_path / STATE_FILENAME, map_location=device or "cpu")
    result.model.load_state_dict(state)
    result.manifest = manifest
    pruner._last_result = result
    return result


def _config_from_dict(config_cls, payload: dict[str, Any]):
    if hasattr(config_cls, "from_dict") and callable(getattr(config_cls, "from_dict")):
        return config_cls.from_dict(payload)
    try:
        return config_cls(**payload)
    except TypeError:
        config = config_cls()
        for key, value in payload.items():
            setattr(config, key, value)
        return config


def _instantiate_model(model_cls, config, device: str | None = None, dtype=None):
    model = model_cls(config)
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model
