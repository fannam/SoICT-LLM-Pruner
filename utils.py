import torch
import torch.nn.functional as F

def calculate_importance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute importance = 1 - mean_cosine_similarity between flattened x and y."""
    # Flatten batch and sequence dims
    flat_x = x.view(-1, x.size(-1)).float()
    flat_y = y.view(-1, y.size(-1)).float()
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(flat_x, flat_y, dim=-1)
    # Replace NaNs (zero vectors) with similarity=1.0 (zero importance)
    cos_sim = torch.nan_to_num(cos_sim, nan=1.0)
    return 1.0 - cos_sim.mean().item()

def calculate_embedding_channels_global_score(embedding_importance):
    all_scores = torch.stack(list(embedding_importance.values()), dim=0)
    global_score = all_scores.sum(dim=0)
    return global_score

class FeedForwardPasser(torch.nn.Module):
    """
    Identity module for MLP layers. Returns hidden_states unmodified.
    (This was likely okay, but defined here for clarity).
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
         return hidden_states

class AttentionPasser(torch.nn.Module):
    """
    Identity module for Attention layers that matches the expected output signature.
    Returns (hidden_states, attention_weights=None).
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        # Return the hidden_states unmodified as the first output.
        # Return None for the attention weights (second expected output).
        return hidden_states, None