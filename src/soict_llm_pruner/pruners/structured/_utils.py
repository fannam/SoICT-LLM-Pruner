from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .types import SliceSpec


def clone_or_share(model, clone_model: bool):
    if clone_model:
        return copy.deepcopy(model)
    return model


def get_module_by_path(root, path: str):
    current = root
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
            continue
        current = getattr(current, part)
    return current


def resolve_slice_tensor(model, slice_spec: SliceSpec) -> torch.Tensor:
    module = get_module_by_path(model, slice_spec.module_path)
    tensor = getattr(module, slice_spec.param_name)
    index = torch.tensor(slice_spec.indices, device=tensor.device, dtype=torch.long)
    return torch.index_select(tensor, dim=slice_spec.axis, index=index)


def copy_tensor_slice(tensor: torch.Tensor, axis: int, indices: list[int]) -> torch.Tensor:
    index = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return torch.index_select(tensor, dim=axis, index=index).clone()


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_causal_lm_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = move_batch_to_device(batch, device)
    if "labels" not in moved:
        if "input_ids" not in moved:
            raise ValueError("Batch must contain either labels or input_ids.")
        moved["labels"] = moved["input_ids"].clone()
    return moved


def compute_causal_lm_loss(outputs, batch: dict[str, Any]) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "loss" in outputs and outputs["loss"] is not None:
            return outputs["loss"]
        logits = outputs["logits"]
    else:
        if getattr(outputs, "loss", None) is not None:
            return outputs.loss
        logits = outputs.logits

    labels = batch["labels"]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
