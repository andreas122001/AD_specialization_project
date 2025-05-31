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
    ):
        super(FocalLoss, self).__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute the focal loss between predictions and targets.

        Args:
            logits: logits for each class, [B, C]
            targets: class targets as a LongTensor, [B]
        """

        # Compute log probs
        log_probs = F.log_softmax(logits, dim=1)
        log_probs = log_probs.gather(1, targets.unsqueeze(1))
        probs = log_probs.exp()

        # Compute focal (1 - probs)^gamma
        focal = (1.0 - probs) ** self.gamma

        if self.weight is None:
            alpha = 1.0
        else:
            alpha = self.weight.to(logits.device)

        # Compute the loss
        loss = -alpha * focal * log_probs
        loss = loss.squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    # Example usage
    loss_fn = FocalLoss(weight=torch.tensor([1.0, 1.0, 1, 1]), gamma=2.0, reduction="mean")
    logits = torch.tensor([[
        [1.0, -1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0, 1.0],
    ]]) * 1  # Example logits for 4 samples and 5 classes
    targets = torch.tensor([[0, 1, 2, 3]])  # Example targets for the 4 samples

    loss = loss_fn(logits, targets)
    print("Focal Loss:", loss.item())
    print("Cross Entropy Loss:", F.cross_entropy(logits, targets).item())


