import torch
import torch.nn.functional as F
import torch.nn as nn

def calculate_importance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute importance = 1 - mean_cosine_similarity between flattened x and y."""
    flat_x = x.view(-1, x.size(-1)).float()
    flat_y = y.view(-1, y.size(-1)).float()
    cos_sim = F.cosine_similarity(flat_x, flat_y, dim=-1)
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
        return hidden_states, None
    
class DistilModel(nn.Module):
    def __init__(self, student, student_dim, teacher_dim, teacher_kept_layers):
        super().__init__()
        self.student = student
        self.teacher_kept_layers = teacher_kept_layers

        if student_dim == teacher_dim:
            self.projectors = nn.ModuleList([nn.Identity() for _ in teacher_kept_layers])
        else:
            self.projectors = nn.ModuleList([
                nn.Linear(student_dim, teacher_dim) for _ in teacher_kept_layers
            ])

    def forward(self, **kwargs):
        return self.student(**kwargs, output_hidden_states=True)
    
class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, **kwargs):
        """
        Passes hidden_states through and returns a tuple compatible with HF DecoderLayer outputs
        when use_cache and output_attentions are False.
        The Qwen2DecoderLayer typically returns (hidden_states, present_key_value, router_logits).
        For non-cached, non-attention output, present_key_value and router_logits can be None.
        """
        # This signature needs to be robust enough for what the rest of the model expects.
        return (hidden_states, None, None) # Matches (hidden_states, present_key_value, router_logits) with None for last two

