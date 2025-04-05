from torch import nn
import torch
from torch.nn.modules.activation import MultiheadAttention


class MHATemporalFusion(nn.Module):
    def __init__(self, embedded_dim, n_heads=1) -> None:
        super().__init__()

        self.embedded_dim = embedded_dim  # 256

        self.cross_attn = MultiheadAttention(embedded_dim, n_heads, 0.2)
        self.self_attn = MultiheadAttention(embedded_dim, n_heads, 0.2)
        self.mlp = nn.Linear(65*embedded_dim, 65*embedded_dim)

    def forward(
        self, current_feature: torch.Tensor, historic_feature: torch.Tensor
    ) -> torch.Tensor:
        

        # Cross-attention
        fused_feature, cross_weights = self.cross_attn(current_feature, historic_feature, historic_feature, return_attention_weights=True)

        # Self-attention
        fused_feature, self_weights = self.self_attn(fused_feature, fused_feature, fused_feature, return_attention_weights=True)

        # # Dense layer
        fused_feature = self.mlp(
            fused_feature.flatten(1)
        ).reshape(-1, 65, self.embedded_dim)  # reshape back to a set of tokens

        return fused_feature, (cross_weights, self_weights)
