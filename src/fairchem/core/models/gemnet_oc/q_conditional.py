import torch
import torch.nn as nn

class QConditionedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        super().__init__()
        self.q_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 使用自定义的cross-attention
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features, q, batch_idx):
        q_embed = self.q_mlp(q)  # [batch_size, hidden_dim]
        q_embed_expanded = q_embed[batch_idx]  # [num_nodes, hidden_dim]
        
        # 将q作为query，node_features作为key和value
        query = self.q_proj(q_embed_expanded)  # [num_nodes, hidden_dim]
        key = self.k_proj(node_features)       # [num_nodes, hidden_dim]
        value = self.v_proj(node_features)     # [num_nodes, hidden_dim]
        
        # 重塑为multi-head格式
        batch_size, num_nodes = node_features.shape[0], node_features.shape[0]
        query = query.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        key = key.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        value = value.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)
        
        # 计算attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用attention
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_nodes, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output + node_features
