from torch import nn
import torch
from torch.nn.modules.activation import MultiheadAttention
from typing import Optional


class MHATemporalFusion(nn.Module):
    def __init__(
        self, embedded_dim: int, hidden_dim: int = 1024, n_heads: int = 1, use_attn_weights: bool = False, learnable_init: bool = True
    ) -> None:
        """
        Args:
            embedded_dim: int - Dimension of the input features (e.g., 256)
            hidden_dim: int - Dimension of the hidden layer in the MLP (e.g., 1024)
            n_heads: int - Number of attention heads (e.g., 1)
            use_attn_weights: bool - Whether to return attention weights, or use optimized attention
            learnable_init: bool - Whether to use a learnable initialization for historic features, else use zeros
        """
        super().__init__()
        self.use_attn_weights = use_attn_weights
        self.embedded_dim = embedded_dim  # e.g. 256

        # Learnable initalization
        self.historic_init = nn.Parameter(torch.zeros(1, 65, 256)) if learnable_init else torch.zeros(1, 65, 256)

        # Fusion layers
        self.cross_attn = MultiheadAttention(embedded_dim, n_heads, dropout=0.2, batch_first=True)
        self.self_attn = MultiheadAttention(embedded_dim, n_heads, dropout=0.2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),  # e.g. 256x1024
            nn.GELU(),  # Default for transformers
            nn.Linear(hidden_dim, embedded_dim)  # e.g. 1024x256
        )


    def forward(
        self, current_feature: torch.Tensor, historic_feature: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            current_feature: [batch_size, n_tokens, n_dim] - Current feature
            historic_feature: [batch_size, n_tokens, n_dim] - Historic feature (optional), will be initialized if None
        Returns:
            fused_feature: [batch_size, n_tokens, n_dim] - The temporally fused feature
            attn_weights: (cross_attn_weights, self_attn_weights), if use_attn_weights is True, else (None, None)
        """


        BZ = current_feature.shape[0]

        # Initialize historic feature
        if historic_feature is None:
            historic_feature = self.historic_init.expand(BZ, -1, -1).to(current_feature.device)

        # Cross-attention
        fused_feature, cross_weights = self.cross_attn(
            current_feature, historic_feature, historic_feature,
            need_weights=self.use_attn_weights, average_attn_weights=True,
        )

        # Self-attention
        fused_feature, self_weights = self.self_attn(
            fused_feature, fused_feature, fused_feature,
            need_weights=self.use_attn_weights, average_attn_weights=True,
        )

        # Token-wise MLP
        fused_feature = self.mlp(fused_feature)

        return fused_feature, (cross_weights, self_weights)
