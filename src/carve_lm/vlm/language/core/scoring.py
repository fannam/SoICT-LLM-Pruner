from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def calculate_importance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute importance as 1 - mean cosine similarity between flattened tensors."""

    flat_x = x.view(-1, x.size(-1)).float()
    flat_y = y.view(-1, y.size(-1)).float()
    cosine_similarity = F.cosine_similarity(flat_x, flat_y, dim=-1)
    cosine_similarity = torch.nan_to_num(cosine_similarity, nan=1.0)
    return 1.0 - cosine_similarity.mean().item()


def calculate_embedding_channels_global_score(
    embedding_importance: Mapping[object, torch.Tensor],
) -> torch.Tensor:
    all_scores = torch.stack(list(embedding_importance.values()), dim=0)
    return all_scores.sum(dim=0)
