from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def clone_or_share(model, clone_model: bool):
    return copy.deepcopy(model) if clone_model else model


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_topk_indices(scores, k: int, *, device: str | torch.device = "cpu") -> list[int]:
    tensor = scores if torch.is_tensor(scores) else torch.as_tensor(scores)
    if tensor.ndim != 1:
        raise ValueError("scores must be a 1D tensor.")
    if not 0 < k <= tensor.numel():
        raise ValueError("k must be in [1, {}], got {}.".format(tensor.numel(), k))
    return torch.sort(torch.topk(tensor.to(device), k=k, largest=True).indices).values.cpu().tolist()


def slice_linear(
    linear: nn.Linear,
    *,
    out_indices: list[int] | None,
    in_indices: list[int] | None,
) -> nn.Linear:
    weight = linear.weight.data
    if out_indices is not None:
        weight = weight[out_indices, :]
    if in_indices is not None:
        weight = weight[:, in_indices]

    new_linear = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=linear.bias is not None,
    ).to(linear.weight.device, dtype=linear.weight.dtype)
    new_linear.weight.data.copy_(weight)
    if linear.bias is not None:
        bias = linear.bias.data
        if out_indices is not None:
            bias = bias[out_indices]
        new_linear.bias.data.copy_(bias)
    return new_linear


def slice_norm(norm: nn.Module, keep_indices: list[int]) -> nn.Module:
    new_norm = copy.deepcopy(norm)
    if hasattr(new_norm, "weight") and getattr(new_norm, "weight") is not None:
        new_norm.weight = nn.Parameter(norm.weight.data[keep_indices].clone())
    if hasattr(new_norm, "bias") and getattr(new_norm, "bias") is not None:
        new_norm.bias = nn.Parameter(norm.bias.data[keep_indices].clone())
    if hasattr(new_norm, "normalized_shape"):
        new_norm.normalized_shape = (len(keep_indices),)
    return new_norm


def slice_projection_output(module: nn.Module, keep_indices: list[int]) -> nn.Module:
    if isinstance(module, nn.Linear):
        return slice_linear(module, out_indices=keep_indices, in_indices=None)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        new_module = copy.deepcopy(module)
        new_module.out_channels = len(keep_indices)
        new_module.weight = nn.Parameter(module.weight.data[keep_indices, ...].clone())
        if module.bias is not None:
            new_module.bias = nn.Parameter(module.bias.data[keep_indices].clone())
        return new_module
    raise TypeError(
        "Unsupported patch projection type {} for output-channel slicing.".format(
            type(module).__name__
        )
    )


def append_pruning_history(model, plan) -> None:
    history = list(getattr(model.config, "pruning_history", []))
    history.append(plan.to_dict())
    setattr(model.config, "pruning_history", history)
    setattr(model.config, "last_pruning_mode", plan.mode)


def config_from_dict(config_cls, payload: dict[str, Any]):
    if hasattr(config_cls, "from_dict") and callable(getattr(config_cls, "from_dict")):
        return config_cls.from_dict(payload)
    try:
        return config_cls(**payload)
    except TypeError:
        config = config_cls()
        for key, value in payload.items():
            setattr(config, key, value)
        return config


def instantiate_model(model_cls, config, device: str | None = None, dtype=None):
    model = model_cls(config)
    if device is not None:
        model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model
