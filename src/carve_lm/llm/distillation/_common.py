from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def get_output_attr(outputs: Any, attr: str) -> Any:
    if isinstance(outputs, Mapping):
        value = outputs.get(attr)
    else:
        value = getattr(outputs, attr, None)
    if value is None:
        raise ValueError(f"Model outputs must provide `{attr}`.")
    return value


def maybe_get_output_attr(outputs: Any, attr: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs.get(attr)
    return getattr(outputs, attr, None)


def resolve_device(model, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    target = torch.device(device)
    return {
        key: value.to(target) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def prepare_causal_lm_batch(
    batch: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    moved = move_batch_to_device(batch, device)
    if "input_ids" not in moved:
        raise ValueError("Batch must contain `input_ids`.")

    input_ids = moved["input_ids"]
    attention_mask = moved.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    labels = moved.get("labels")
    if labels is None:
        labels = input_ids.clone()

    shifted_labels = labels[..., 1:].contiguous().long()
    loss_mask = attention_mask[..., 1:].contiguous().bool()
    if shifted_labels.eq(-100).any():
        loss_mask = loss_mask & shifted_labels.ne(-100)

    return input_ids, attention_mask, shifted_labels, loss_mask


def masked_logits_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student logits must have the same shape.")
    if temperature <= 0:
        raise ValueError("`temperature` must be > 0.")
    if not 0 <= alpha <= 1:
        raise ValueError("`alpha` must be between 0 and 1.")

    flat_student = student_logits.reshape(-1, student_logits.size(-1))
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_mask = loss_mask.reshape(-1).bool()

    if flat_student.size(0) != flat_mask.numel() or flat_labels.numel() != flat_mask.numel():
        raise ValueError("Labels and mask must align with the logits sequence length.")
    if not torch.any(flat_mask):
        raise ValueError("Loss mask does not contain any valid tokens.")

    masked_student = flat_student[flat_mask]
    masked_teacher = flat_teacher[flat_mask]
    masked_labels = flat_labels[flat_mask]

    log_student = F.log_softmax(masked_student / temperature, dim=-1)
    soft_teacher = F.softmax(masked_teacher / temperature, dim=-1)
    forward_kl = F.kl_div(log_student, soft_teacher, reduction="batchmean") * (temperature**2)
    ce_loss = F.cross_entropy(masked_student, masked_labels)
    return alpha * ce_loss + (1 - alpha) * forward_kl


def masked_feature_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
    projectors,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(loss_mask):
        raise ValueError("Feature loss mask does not contain any valid tokens.")

    losses = []
    for proj, s_feat, t_feat in zip(projectors, student_feats, teacher_feats):
        projected = proj(s_feat.float())[loss_mask]
        target = t_feat.float()[loss_mask]
        losses.append(F.mse_loss(projected, target, reduction="mean"))
    if not losses:
        raise ValueError("At least one feature pair is required for hybrid distillation.")
    return torch.stack(losses).mean()


def zero_grad(optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def maybe_init_wandb(
    *,
    enabled: bool,
    project_name: str | None,
    run_name: str | None,
    wandb_key: str | None,
):
    if not enabled:
        return None

    import wandb

    if wandb_key:
        wandb.login(key=wandb_key)
    return wandb.init(project=project_name, name=run_name, reinit=True)


def maybe_log_wandb(run, payload: dict[str, Any]) -> None:
    if run is not None:
        run.log(payload)


def maybe_finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def plot_history(
    values: list[float],
    *,
    title: str,
    ylabel: str,
    show: bool,
):
    if not values:
        return None

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Plotting requires `matplotlib`. Install the `train` extra first.") from exc

    fig, ax = plt.subplots()
    ax.plot(range(1, len(values) + 1), values, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if show:
        plt.show()
    return fig


def sync_scheduler_added_param_group_lrs(optimizer, scheduler) -> None:
    if scheduler is None or not hasattr(scheduler, "base_lrs"):
        return
    managed_groups = len(scheduler.base_lrs)
    if len(optimizer.param_groups) <= managed_groups:
        return

    reference_lr = optimizer.param_groups[0]["lr"]
    for param_group in optimizer.param_groups[managed_groups:]:
        param_group["lr"] = reference_lr
