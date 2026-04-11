from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OTConfig:
    weight: float = 0.1
    epsilon: float = 0.05
    sinkhorn_iters: int = 20
    position_weight: float = 0.05
    window_radius: int = 16
    max_tokens_per_sequence: int = 128

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("`weight` must be >= 0.")
        if self.epsilon <= 0:
            raise ValueError("`epsilon` must be > 0.")
        if self.sinkhorn_iters < 1:
            raise ValueError("`sinkhorn_iters` must be >= 1.")
        if self.position_weight < 0:
            raise ValueError("`position_weight` must be >= 0.")
        if self.window_radius < 0:
            raise ValueError("`window_radius` must be >= 0.")
        if self.max_tokens_per_sequence < 1:
            raise ValueError("`max_tokens_per_sequence` must be >= 1.")


def _evenly_spaced_indices(length: int, target_length: int, device: torch.device) -> torch.Tensor:
    if target_length >= length:
        return torch.arange(length, device=device, dtype=torch.long)
    if target_length == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    return torch.linspace(
        0,
        length - 1,
        steps=target_length,
        device=device,
        dtype=torch.float32,
    ).round().to(dtype=torch.long)


def _position_cost(num_x: int, num_y: int, device: torch.device) -> torch.Tensor:
    scale_x = max(num_x - 1, 1)
    scale_y = max(num_y - 1, 1)
    pos_x = torch.arange(num_x, device=device, dtype=torch.float32) / float(scale_x)
    pos_y = torch.arange(num_y, device=device, dtype=torch.float32) / float(scale_y)
    return (pos_x[:, None] - pos_y[None, :]).pow(2)


def _cost_and_band_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    position_weight: float,
    window_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm = F.normalize(x, dim=-1, eps=1e-12)
    y_norm = F.normalize(y, dim=-1, eps=1e-12)
    cosine_distance = 1.0 - torch.matmul(x_norm, y_norm.transpose(0, 1))
    cost = cosine_distance + position_weight * _position_cost(x.size(0), y.size(0), x.device)

    ordinal_x = torch.arange(x.size(0), device=x.device, dtype=torch.long)
    ordinal_y = torch.arange(y.size(0), device=x.device, dtype=torch.long)
    band_mask = torch.abs(ordinal_x[:, None] - ordinal_y[None, :]) <= window_radius
    return cost, band_mask


def sinkhorn_transport_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: OTConfig | None = None,
) -> torch.Tensor:
    ot_config = config or OTConfig()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("sinkhorn_transport_cost expects 2D tensors shaped [tokens, width].")
    if x.size(0) == 0 or y.size(0) == 0:
        raise ValueError("sinkhorn_transport_cost requires at least one token per side.")

    x_fp32 = x.to(device=x.device, dtype=torch.float32)
    y_fp32 = y.to(device=x.device, dtype=torch.float32)
    cost, band_mask = _cost_and_band_mask(
        x_fp32,
        y_fp32,
        position_weight=ot_config.position_weight,
        window_radius=ot_config.window_radius,
    )
    if not torch.all(band_mask.any(dim=1)) or not torch.all(band_mask.any(dim=0)):
        raise ValueError("Locality band produced an infeasible transport problem.")

    safe_cost = cost.masked_fill(~band_mask, 0.0)
    log_kernel = torch.full_like(cost, -torch.inf)
    log_kernel[band_mask] = -cost[band_mask] / float(ot_config.epsilon)

    log_u = torch.zeros(x_fp32.size(0), device=x.device, dtype=torch.float32)
    log_v = torch.zeros(y_fp32.size(0), device=x.device, dtype=torch.float32)
    log_a = torch.full_like(log_u, -math.log(float(x_fp32.size(0))))
    log_b = torch.full_like(log_v, -math.log(float(y_fp32.size(0))))

    for _ in range(ot_config.sinkhorn_iters):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)

    log_pi = log_u[:, None] + log_kernel + log_v[None, :]
    transport_plan = torch.exp(log_pi)
    return (transport_plan * safe_cost).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: OTConfig | None = None,
) -> torch.Tensor:
    ot_config = config or OTConfig()
    xy_cost = sinkhorn_transport_cost(x, y, config=ot_config)
    xx_cost = sinkhorn_transport_cost(x, x, config=ot_config)
    yy_cost = sinkhorn_transport_cost(y, y, config=ot_config)
    return xy_cost - 0.5 * xx_cost - 0.5 * yy_cost


def masked_ot_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
    projectors,
    loss_mask: torch.Tensor,
    ot_config: OTConfig,
) -> torch.Tensor:
    if not torch.any(loss_mask):
        raise ValueError("OT loss mask does not contain any valid tokens.")

    losses = []
    for proj, s_feat, t_feat in zip(projectors, student_feats, teacher_feats):
        projected = proj(s_feat.float())
        target = t_feat.detach().to(device=projected.device, dtype=torch.float32)
        for batch_idx in range(loss_mask.size(0)):
            valid_indices = torch.nonzero(loss_mask[batch_idx], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            if valid_indices.numel() > ot_config.max_tokens_per_sequence:
                sample_indices = _evenly_spaced_indices(
                    valid_indices.numel(),
                    ot_config.max_tokens_per_sequence,
                    valid_indices.device,
                )
                valid_indices = valid_indices[sample_indices]

            student_sequence = projected[batch_idx].index_select(0, valid_indices)
            teacher_sequence = target[batch_idx].index_select(0, valid_indices)
            losses.append(
                sinkhorn_divergence(
                    student_sequence,
                    teacher_sequence,
                    config=ot_config,
                )
            )

    if not losses:
        raise ValueError("At least one valid sequence is required for OT distillation.")
    return torch.stack(losses).mean()


__all__ = [
    "OTConfig",
    "masked_ot_loss",
    "sinkhorn_divergence",
    "sinkhorn_transport_cost",
]
