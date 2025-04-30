import torch
from torch import nn
from typing import Optional, Literal
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
    ):
        super(FocalLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute the focal loss between predictions and targets.

        Args:
            logits: logits for each class, [B, C]
            targets: class targets as a LongTensor, [B]
        """
        # Apply label smoothing if specified
        num_classes = logits.shape[1]
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smoothed_targets = (1.0 - self.label_smoothing) * targets_one_hot + self.label_smoothing / num_classes

        # Compute log probs
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Compute focal (1 - probs)^gamma
        focal = (1.0 - probs) ** self.gamma

        if self.weight is None:
            alpha = torch.ones(num_classes, device=logits.device)
        else:
            if self.weight.dim() != 1 or self.weight.size(0) != num_classes:
                raise ValueError(f"weight must be a tensor of shape [num_classes={num_classes}]")
            alpha = self.weight.to(logits.device)

        # Compute the loss
        loss = alpha * focal * smoothed_targets * (-log_probs)
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    # Example usage
    loss_fn = FocalLoss(weight=torch.tensor([0.1, 1.9, 1, 1]), gamma=2.0, reduction="mean")
    logits = torch.tensor([
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, 1.0],
    ]) * 2  # Example logits for 4 samples and 5 classes
    targets = torch.tensor([0, 1, 2, 3])  # Example targets for the 4 samples
    loss = loss_fn(logits, targets)
    print("Focal Loss:", loss.item())

