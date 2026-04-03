from __future__ import annotations

import torch.nn as nn


class FeedForwardPasser(nn.Module):
    """Identity module for MLP layers."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class AttentionPasser(nn.Module):
    """Identity attention module that preserves the common HF return shape."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        return hidden_states, None


class IdentityLayer(nn.Module):
    """Identity decoder layer compatible with decoder-only HF blocks."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        return hidden_states, None, None
