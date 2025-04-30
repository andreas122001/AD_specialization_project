from torch import nn
import torch
from torch.nn.modules.activation import MultiheadAttention
from typing import Optional


class TemporalFusionBlock(nn.Module):
    def __init__(
        self, embedded_dim: int, hidden_dim: int = 1024, n_heads: int = 1, dropout: float = 0.2, use_attn_weights: bool = False
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

        # Fusion layers
        self.cross_attn = MultiheadAttention(embedded_dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = MultiheadAttention(embedded_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),  # e.g. 256x1024
            nn.GELU(),  # Default for transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedded_dim),  # e.g. 1024x256
            nn.Dropout(dropout),
        )
        self.historic_norm = nn.LayerNorm(embedded_dim)
        self.cross_norm = nn.LayerNorm(embedded_dim)
        self.self_norm = nn.LayerNorm(embedded_dim)


    def forward(
        self, current_feature: torch.Tensor, historic_feature: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            current_feature: [batch_size, n_tokens, n_dim] - Current feature
            historic_feature: [batch_size, n_tokens, n_dim] - Historic feature (optional), will be initialized if None
        Returns:
            fused_feature: [batch_size, n_tokens, n_dim] - The temporally fused feature
            attn_weights: (cross_attn_weights, self_attn_weights), if use_attn_weights is True, else (None, None)
        """

        # Need to normalize the historic feat.,
        # current feat. is already normalized in TF backbone output
        historic_feature = self.historic_norm(historic_feature)

        # Cross-attention, fuse current and historic features
        fused_feature, cross_weights = self.cross_attn(
            current_feature, historic_feature, historic_feature,
            need_weights=self.use_attn_weights, average_attn_weights=True,
        )
        fused_feature = current_feature + fused_feature  # residual
        fused_feature = self.cross_norm(fused_feature)

        # Self-attention on fused features
        fused_feature_tmp, self_weights = self.self_attn(
            fused_feature, fused_feature, fused_feature,
            need_weights=self.use_attn_weights, average_attn_weights=True,
        )
        fused_feature = fused_feature_tmp + fused_feature  # residual
        fused_feature = self.self_norm(fused_feature)

        # Token-wise MLP
        fused_feature_tmp = self.mlp(fused_feature)
        fused_feature = fused_feature_tmp + fused_feature  # residual

        return fused_feature, (cross_weights, self_weights)


class TemporalFusionBlockNoSelfAttn(nn.Module):
    def __init__(
        self, embedded_dim: int, hidden_dim: int = 1024, n_heads: int = 1, dropout: float = 0.2, use_attn_weights: bool = False
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

        # Fusion layers
        self.cross_attn = MultiheadAttention(embedded_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),  # e.g. 256x1024
            nn.GELU(),  # Default for transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedded_dim),  # e.g. 1024x256
            nn.Dropout(dropout),
        )
        self.historic_norm = nn.LayerNorm(embedded_dim)
        self.cross_norm = nn.LayerNorm(embedded_dim)


    def forward(
        self, current_feature: torch.Tensor, historic_feature: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            current_feature: [batch_size, n_tokens, n_dim] - Current feature
            historic_feature: [batch_size, n_tokens, n_dim] - Historic feature (optional), will be initialized if None
        Returns:
            fused_feature: [batch_size, n_tokens, n_dim] - The temporally fused feature
            attn_weights: (cross_attn_weights, None), if use_attn_weights is True, else (None, None), (None instead of the self-attn weights for compatibility)
        """

        # Need to normalize the historic feat.,
        # current feat. is already normalized in TF backbone output
        historic_feature = self.historic_norm(historic_feature)

        # Cross-attention, fuse current and historic features
        fused_feature, cross_weights = self.cross_attn(
            current_feature, historic_feature, historic_feature,
            need_weights=self.use_attn_weights, average_attn_weights=True,
        )
        fused_feature = current_feature + fused_feature  # residual
        fused_feature = self.cross_norm(fused_feature)

        # Token-wise MLP
        fused_feature_tmp = self.mlp(fused_feature)
        fused_feature = fused_feature_tmp + fused_feature  # residual

        return fused_feature, (cross_weights, None)


class MHATemporalFusion(nn.Module):
    def __init__(
        self, embedded_dim: int, n_layers: int = 1, hidden_dim: int = 1024, n_heads: int = 1, dropout: float = 0.2, use_attn_weights: bool = False, learnable_init: bool = True, use_self_attn = True
    ) -> None:
        """
        Args:
            embedded_dim: int - Dimension of the input features (e.g., 256)
            n_layers: int - How many temporal fusion blocks to use
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

        # Set the block class based on whether self-attention is used
        block_cls = TemporalFusionBlock if self_attn else TemporalFusionBlockNoSelfAttn

        # Fusion layers
        self.layers = nn.ModuleList(
            [block_cls(
                embedded_dim=embedded_dim, 
                hidden_dim=hidden_dim, 
                n_heads=n_heads, 
                dropout=dropout, 
                use_attn_weights=use_attn_weights,
            ) for _ in range(n_layers)]
        )

    def forward(
        self, current_feature: torch.Tensor, historic_feature: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            current_feature: [batch_size, n_tokens, n_dim] - Current feature
            historic_feature: [batch_size, n_tokens, n_dim] - Historic feature (optional), will be initialized if None
        Returns:
            fused_feature: [batch_size, n_tokens, n_dim] - The temporally fused feature
            attn_weights: list[(cross_attn_weights, self_attn_weights)], if use_attn_weights is True, else list[(None, None)]
        """

        BZ = current_feature.shape[0]
        attention_weights: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Initialize historic feature
        if historic_feature is None:
            historic_feature = self.historic_init.expand(BZ, -1, -1).to(current_feature.device)

        # Process through fusion layers
        for layer in self.layers:
            current_feature, (cross_weights, self_weights) = layer(current_feature, historic_feature)
            attention_weights.append((cross_weights, self_weights))

        return current_feature, attention_weights

