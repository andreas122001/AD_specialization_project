import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional
from torch import nn
import torch.nn.functional as F

class MultiModalHungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.trajectory_loss = nn.L1Loss(reduction="none")
        self.confidence_loss = nn.CrossEntropyLoss()

    def forward(self, 
            outputs: tuple[torch.Tensor, torch.Tensor],
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ):
        """
        Compute the Hungarian loss for predicted 2D multimodal trajectories with batch B, number of objects N, modes K, and future timesteps F.

        Args:
            outputs: tuple containing predicted trajectories and predicted confidence scores.
                    - pred_trajectory: Tensor of shape [B, N, K, F, 2]
                    - pred_confidence: Tensor of shape [B, N, K+1] (logits for mode/background classes)
            targets: padded ground truth trajectories [B, N, F, 2].
                    Valid trajectories have non-zero values, while padded ones are zeros.
            mask: trajectory mask for the GT trajectories [B, N, F].
                    Useful for ignoring parts of GT trajectories if some points are missing.
        """
        pred_trajectories, pred_confidences = outputs  # [B, N, K, F, 2], [B, N, K+1]
        batch_size, n_queries, n_modes, F, _ = pred_trajectories.shape
        
        # Compute a per-trajectory validity flag for ground truth.
        # Here, if the sum of absolute values in a trajectory is not zero, it is considered valid.
        # Otherwise, it is most certainly just padding.
        valid_indices = targets.abs().sum(dim=(2, 3)).not_equal(0).long()  # Shape: [B, N]

        # Mask out invalid trajectory points
        # if some points are missing (e.g. actor is missing for some frame), we exclude them
        # We don't want to penalize the model for bad GTs
        if mask is not None:
            targets = targets * mask.unsqueeze(-1)

        # Initialize the target confidences, which will contain the reordered valid targets (after matching)
        target_confidences = torch.ones(
            batch_size, n_queries, device=pred_trajectories.device
        ).long() * n_modes  # initialize to last class (background class)

        total_traj_loss = 0.0

        for b in range(batch_size):
            pred_traj = pred_trajectories[b]  # [N, K, F, 2]
            targets_b = targets[b]  # [N, F, 2]
            valid_mask_b = valid_indices[b].bool()  # [N]
            valid_targets = targets_b[valid_mask_b]  # [n_valid, F, 2]
            n_valid = valid_targets.shape[0]

            # If no valid targets exist, continue. 
            # We don't need to match it since they are all zero anyway.
            if n_valid == 0:
                continue

            # Expand dimensions for broadcasting
            # Predicted trajectories are copied across all valid target trajectories
            # Targets are copied across all predicted trajectory modes
            pred_exp = pred_traj.unsqueeze(2).expand(-1, -1, n_valid, -1, -1)  # [N, K, n, F, 2]
            tgt_exp = valid_targets.unsqueeze(0).unsqueeze(0).expand(pred_traj.shape[0], n_modes, -1, -1, -1)  # [N, K, n, F, 2]
            
            # Vectorized multimodal cost matrix -> [N, K, n_valid]
            # Ignore the fact that some targets are masked...
            # Since they are unmatched, we don't know which mask to apply (makes it hard to vectorize)
            cost_matrix = self.trajectory_loss(pred_exp, tgt_exp).mean(dim=(3,4))

            # Find best mode per query-target pair to get [N, n]
            min_cost, best_modes = cost_matrix.min(dim=1)  # [N, n]

            # Hungarian matching on minimal cost modes (detached since its non-differentiable)
            with torch.no_grad():
                cost_np = min_cost.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Get the best mode for each target-pred pair
            best_mode = best_modes[row_idx, col_idx] 

            # Get the matched pairs
            matched_preds = pred_traj[row_idx, best_mode]  # [n_valid, F, 2]
            matched_targets = valid_targets[col_idx]  # [n_valid, F, 2]

            # Overwrite backgrounds with mode index (as class)
            target_confidences[b, row_idx] = best_mode

            # Need to mask the matched preds as well
            if mask is not None:
                original_indices = torch.where(valid_mask_b)[0]
                mask_index = original_indices[col_idx]
                matched_preds = matched_preds * mask[b][mask_index].unsqueeze(-1)

            # Recalculate loss with masked preds given matched targets
            total_traj_loss += self.trajectory_loss(matched_preds, matched_targets).sum().divide(n_valid)

        # Confidence loss
        avg_traj_loss = total_traj_loss / batch_size
        avg_conf_loss = self.confidence_loss(
            pred_confidences.view(-1, n_modes + 1),  # [B*N, K+1]
            target_confidences.view(-1)  # [B*N]
        )

        return {
            "loss_trajectories": avg_traj_loss,
            "loss_trajectory_confidence": avg_conf_loss,
        }


if __name__ == "__main__":
    # Debugging
    B, N, K, F = 1, 4, 32, 10

    pred_trajectory = torch.rand(B, N, K, F, 2)*10
    #              [B,N,K,2]
    pred_trajectory[:, 0, 0, :] = torch.tensor([1.0, 1.0])# + 0.001  # 0->1
    pred_trajectory[:, 1, 1, :] = torch.tensor([1.0, 2.0])# + 0.001  # 1->0
    pred_trajectory[:, 2, 0, :] = torch.tensor([2.0, 1.0])# + 0.001  # 2->2

    pred_confidence = torch.zeros(B, N, K + 1) - torch.ones(K + 1) * 10
    #              [B,N,K]
    pred_confidence[:, 0, 0] *= -1
    pred_confidence[:, 1, 1] *= -1
    pred_confidence[:, 2, 0] *= -1
    pred_confidence[:, 3, K] *= -1  # background

    targets = torch.zeros(B, N, F, 2)
    targets[:, 0, :] = torch.tensor([1.0, 2.0])
    targets[:, 1, :] = torch.tensor([1.0, 1.0])
    targets[:, 2, :] = torch.tensor([2.0, 1.0])
    # targets[:,2,:] = torch.tensor([2.0, 1.0])

    mask = torch.ones(B, N, F)
    mask[-1] = 0  # last query not used

    mask[:,0,1] = 0
    mask[:,2,0] = 0


    criterion = MultiModalHungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    losses = criterion(outputs, targets, mask)
    print(losses)
