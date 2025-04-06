import torch
from torch import nn
import torch.nn.functional as F

"""
Trajectory decoder for non-ego trajectory prediction.
Based on VAD's Motion Transformer implementation (see https://github.com/hustvl/VAD).
"""

class TrajectoryDecoderLayer(nn.Module):
    def __init__(self, n_dim=256, n_heads=4, hidden_dim=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(n_dim, n_heads, dropout=0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(n_dim, n_heads, dropout=0.1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_dim),
        )

        self.query_norm = nn.LayerNorm(n_dim)
        self.features_norm = nn.LayerNorm(n_dim)
        self.cross_norm = nn.LayerNorm(n_dim)

    def forward(self, query, features):

        # Need to normalize, since temporal module does not
        query = self.query_norm(query)
        features = self.features_norm(features)

        # Cross attention, vehicle queries attending to temporal features
        attn_output, _ = self.cross_attn(query, features, features, need_weights=False)
        query = query + attn_output  # residual
        query = self.cross_norm(query)  # post-cross attention layer norm

        # Token-wise dense layer, 
        ff_output = self.mlp(query)
        query = query + ff_output  # residual

        return query


class TrajectoryDecoder(nn.Module):
    def __init__(self, n_dim=256, n_layers=1, n_queries=10, future_steps=6):
        super().__init__()
        self.future_steps = future_steps

        self.query_embed = nn.Embedding(n_queries, n_dim)
        self.layers = nn.ModuleList(
            [TrajectoryDecoderLayer(n_dim) for _ in range(n_layers)]
        )

        # Prediction heads
        self.traj_head = nn.Linear(n_dim, future_steps * 2)  # (x,y) per timestep
        self.confidence_head = nn.Linear(n_dim, 2)  # binary classification for object/no-object

        nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)

    def forward(self, temporal_features):
        # temporal_features: (BZ, 65, 256)
        batch_size = temporal_features.shape[0]

        # Initialize learnable queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(
            batch_size, 1, 1  # (BZ, N, 256)
        )

        # Process through decoder layers
        for layer in self.layers:
            queries = layer(queries, temporal_features)

        # Final predictions
        traj = self.traj_head(queries)  # (BZ, N, future_steps*2)
        traj = traj.reshape(batch_size, -1, self.future_steps, 2)  # (BZ, N, future_steps, 2)
        confidence_logits = self.confidence_head(queries).squeeze(-1)  # (BZ, N, 2)

        return traj, confidence_logits
