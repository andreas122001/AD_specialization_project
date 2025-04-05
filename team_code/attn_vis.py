import matplotlib.pyplot as plt
import numpy as np

"""
Some code snippets from AI for attn. visualization, not sure how to use them yet, but could be useful.
Some if these work, and some of them don't, I'll filter them later.
"""


def plot_attention(
    attention_weights,
    title="Attention Map",
    xlabel="Source Tokens",
    ylabel="Target Tokens",
):
    """
    attention_weights: [n_heads, target_len, source_len] or [target_len, source_len]
    """
    if attention_weights.dim() == 3:
        # Average across attention heads
        attn = attention_weights.mean(dim=0).cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(attn, cmap="viridis", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Highlight ego token position (index 64)
    plt.axvline(x=64, color="r", linestyle="--", linewidth=1)
    plt.axhline(y=64, color="r", linestyle="--", linewidth=1)

    plt.show()


def plot_token_flow(weights_list, token_idx=64):
    """Track attention to ego token across layers"""
    plt.figure(figsize=(12, 6))

    # Cross-attention to historic
    plt.subplot(121)
    plt.plot(weights_list[0][:, token_idx].mean(dim=0), label="Cross-Attn to Historic")

    # Self-attention
    plt.subplot(122)
    plt.plot(weights_list[1][:, token_idx].mean(dim=0), label="Self-Attn in Fused")

    plt.suptitle(f"Attention Flow for Ego Token (Index {token_idx})")
    plt.legend()
    plt.show()



def attention_rollout(attentions, head_reduction="mean"):
    """Combine attention across layers"""
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        attn = attn.mean(dim=1) if head_reduction == "mean" else attn.sum(dim=1)
        result = torch.matmul(attn, result)
    return result


# Usage
rollout = attention_rollout([cross_weights[1], self_weights[1]])
plot_attention(rollout, title="Attention Rollout")


# 1. Temporal Contribution Ratio
def compute_temporal_contribution(cross_attn_weights):
    """
    cross_attn_weights: [n_heads, n_current_tokens, n_historic_tokens]
    Returns: (historic_ratio, current_ratio)
    """
    # Attention to history is direct sum
    historic_contribution = cross_attn_weights.mean(
        dim=(0, 1)
    )  # Avg over heads & queries

    # Attention to "current" requires comparing residual connections
    residual_strength = torch.norm(fused_features - cross_attn_output) / torch.norm(
        cross_attn_output
    )
    current_ratio = 1 - historic_contribution

    return historic_contribution.mean().item(), residual_strength.item()


def plot_temporal_flow(attn_weights, historic_labels, current_labels):
    plt.figure(figsize=(12, 6))

    # Historic contribution per token
    plt.subplot(121)
    plt.bar(range(len(historic_labels)), attn_weights.mean((0, 1)))
    plt.xticks(ticks=range(len(historic_labels)), labels=historic_labels, rotation=90)
    plt.title("Historic Token Contribution")

    # Current token dependency
    plt.subplot(122)
    plt.barh(range(len(current_labels)), attn_weights.mean((0, 2)))
    plt.yticks(ticks=range(len(current_labels)), labels=current_labels)
    plt.title("Current Token Historical Reliance")

    plt.tight_layout()


# Example: For a driving scenario
historic_labels = [f"Historic_{i}" for i in range(64)] + ["Historic_Ego"]
current_labels = [f"Current_{i}" for i in range(64)] + ["Current_Ego"]

plot_temporal_flow(cross_weights[1], historic_labels, current_labels)


def generate_scoreboard(cross_weights, self_weights):
    # Cross-attention stats
    historic_ratio = cross_weights.mean().item()
    ego_historic_attn = cross_weights[..., 64].mean().item()  # Ego token index

    # Self-attention stats
    self_entropy = -torch.sum(
        self_weights * torch.log(self_weights + 1e-9), dim=-1
    ).mean()

    print(
        f"""
    Temporal Fusion Report
    ----------------------------------
    1. Cross-Attention:
       - Historic contribution: {historic_ratio:.1%}
       - Ego token attention:   {ego_historic_attn:.1%}
    
    2. Self-Attention:
       - Information mixing:    {self_entropy:.3f} nats
    """
    )


# Advanced Technique: Temporal Attribution Mapping
def temporal_attribution(inputs, model):
    # Register hooks
    attn_maps = []

    def hook(module, input, output):
        attn_maps.append(output[1].detach())  # Store attention weights

    handle = model.temporal_fusion.cross_attn.register_forward_hook(hook)

    # Run inference
    with torch.no_grad():
        model(inputs)

    handle.remove()

    # Process attention
    attn = torch.stack(attn_maps).mean(0)  # Avg across layers

    # Visualize
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.xlabel("Historic Tokens")
    plt.ylabel("Current Tokens")
    plt.title("Temporal Attribution Map")
