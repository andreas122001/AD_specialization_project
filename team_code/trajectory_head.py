import torch
from torch import nn
import torch.nn.functional as F

"""
Motion Transformer for trajectory prediction.
Based on VAD's implementation (see https://github.com/hustvl/VAD).
"""

class MotionTransformerDecoderLayer(nn.Module):
    def __init__(self, n_dim=256, n_heads=4, hidden_dim=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(n_dim, n_heads, dropout=0.1, batch_first=True)
        self.self_attn = nn.MultiheadAttention(n_dim, n_heads, dropout=0.1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.GeLU(),
            nn.Linear(hidden_dim, n_dim)
        )

        self.norm1 = nn.LayerNorm(n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.norm3 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, features):
        # Cross attention: query attends to features (temporal features)
        attn_output, _ = self.cross_attn(query, features, features)
        query = query + self.dropout(attn_output)
        query = self.norm1(query)

        # Self attention for temporal coherence
        self_attn_output, _ = self.self_attn(query, query, query)
        query = query + self.dropout(self_attn_output)
        query = self.norm2(query)

        # FFN
        ff_output = self.mlp(query)
        query = query + self.dropout(ff_output)
        query = self.norm3(query)

        return query


class MotionTransformer(nn.Module):
    def __init__(self, n_dim=256, n_layers=1, n_queries=10, future_steps=6):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, n_dim)
        self.layers = nn.ModuleList(
            [MotionTransformerDecoderLayer(n_dim) for _ in range(n_layers)]
        )

        # Prediction heads
        self.traj_head = nn.Linear(n_dim, future_steps * 2)  # (x,y) per timestep
        self.confidence_head = nn.Linear(n_dim, 1)

        nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

    def forward(self, temporal_features):
        # temporal_features: [B, 65, 256]
        batch_size = temporal_features.shape[0]

        # Initialize learnable queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(
            batch_size, 1, 1  # [B, N, 256]
        )

        # Process through decoder layers
        for layer in self.layers:
            queries = layer(queries, temporal_features)

        # Final predictions
        queries = queries.permute(1, 0, 2)  # [B, N, 256]
        traj = self.traj_head(queries)  # [B, N, future_steps*2]
        confidence = self.confidence_head(queries).squeeze(-1)  # [B, N]

        return traj.reshape(batch_size, -1, 6, 2), torch.sigmoid(confidence)
