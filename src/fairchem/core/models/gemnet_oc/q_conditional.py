import torch
import torch.nn as nn

class QConditionedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        super().__init__()
        self.q_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

    def forward(self, node_features, q, batch_idx):
        q_embed = self.q_mlp(q)
        q_embed_expanded = q_embed[batch_idx]
        attn_output, _ = self.attention(
            q_embed_expanded.unsqueeze(0),
            node_features.unsqueeze(0),
            node_features.unsqueeze(0)
        )
        return attn_output.squeeze(0) + node_features
