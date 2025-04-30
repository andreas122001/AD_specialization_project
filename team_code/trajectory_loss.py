import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional
from torch import nn
from focal_loss2 import FocalLoss

class MultiModalHungarianLoss(nn.Module):
    def __init__(self, loss_type="huber"):
        super().__init__()
        if loss_type == "l1":
            self.trajectory_loss = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.trajectory_loss = nn.MSELoss(reduction="none")
        elif loss_type == "huber":
            self.trajectory_loss = nn.HuberLoss(reduction="none", delta=1.0)
        else:
            raise ValueError(f"Unknown loss type for trajectory task: {loss_type}. Use 'l1', 'l2', or 'huber'.")
        self.confidence_loss = FocalLoss(gamma=2.0, label_smoothing=0.01, reduction="mean")

    def forward(self, 
            predictions: tuple[torch.Tensor, torch.Tensor],
            targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ):
        """
        Compute the Hungarian loss for predicted 2D multimodal trajectories with batch B, number of objects N, modes K, and future timesteps T.

        Args:
            predictions: tuple containing predicted trajectories and predicted confidence scores.
                    - pred_trajectory: Tensor of shape [B, N, K, T, 2]
                    - pred_confidence: Tensor of shape [B, N, K+1] (logits for mode/background classes)
            targets: padded ground truth trajectories [B, N, T, 2].
                    Valid trajectories have non-zero values, while padded ones are zeros.
            mask: trajectory mask for the GT trajectories [B, N, T].
                    Useful for ignoring parts of GT trajectories if some points are missing.
        """
        pred_trajectories, pred_confidences = predictions  # [B, N, K, T, 2], [B, N, K+1]
        batch_size, n_queries, n_modes, T, coord_dim = pred_trajectories.shape

        # Mask out invalid trajectory points
        # if some points are missing (e.g. actor is missing for some frame), we exclude them
        # We don't want to penalize the model for bad GTs
        if mask is not None:
            targets = targets * mask.unsqueeze(-1)

        # Compute a per-trajectory validity flag for ground truth.
        # Here, if the sum of absolute values in a trajectory is not zero, it is considered valid.
        # Otherwise, it is most certainly just padding or fully masked out.
        valid_indices = targets.abs().sum(dim=(2, 3)).not_equal(0)  # Shape: [B, N]

        # Initialize the target confidences, which will contain the reordered valid targets (after matching)
        target_confidences = torch.ones(
            batch_size, n_queries, device=pred_trajectories.device
        ).long() * n_modes  # initialize to last class (background class)

        total_traj_loss = torch.zeros((), device=pred_trajectories.device)

        valid_batches = batch_size
        for b in range(batch_size):
            pred_traj = pred_trajectories[b]  # [N, K, T, 2]
            targets_b = targets[b]  # [N, T, 2]
            valid_indices_b = valid_indices[b]  # [N]
            valid_targets = targets_b[valid_indices_b]  # [n_valid, T, 2]
            n_valid = valid_targets.shape[0]

            # If no valid targets exist, continue. 
            # We don't need to match it since they are all zero anyway.
            if n_valid == 0:
                valid_batches -= 1
                continue

            # Expand dimensions for broadcasting
            # Predicted trajectories are copied across all valid target trajectories
            # Targets are copied across all predicted trajectory modes
            pred_exp = pred_traj.unsqueeze(2).expand(-1, -1, n_valid, -1, -1)  # [N, K, n, T, 2]
            tgt_exp = valid_targets.unsqueeze(0).unsqueeze(0).expand(pred_traj.shape[0], n_modes, -1, -1, -1)  # [N, K, n, T, 2]

            if mask is not None:
                # mask pred also (target is already masked)
                mask_exp = mask[b][valid_indices_b].reshape(1, 1, n_valid, T, 1)
                pred_exp = pred_exp  *mask_exp  # [N, K, n, T, 2]

            # Define a normalization factor based on the valid target points n*T*2
            # we know mask is > 0 (since n_valid > 0)
            norm = mask[b, valid_indices_b].sum(dim=1) * coord_dim if mask is not None \
                    else torch.tensor(
                            [T * coord_dim], 
                            device=pred_exp.device
                        ) # if mask is None, divide by total num points
            norm = norm.view(1,1,-1).to(pred_exp.dtype)  # [1, 1, n_valid]
            
            # Vectorized multimodal cost matrix -> [N, K, n_valid]
            # Average over only valid points (masked points should not contribute to loss)
            cost_matrix = self.trajectory_loss(pred_exp, tgt_exp) \
                .sum(dim=(3,4)) \
                .divide(norm)  # [N, K, n_valid]

            # Find best mode per query-target pair to get [N, n]
            min_cost, best_modes = cost_matrix.min(dim=1)  # [N, n]

            # Hungarian matching on minimal cost modes (detached since its non-differentiable)
            with torch.no_grad():
                cost_np = min_cost.detach().cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Get the best mode for each target-pred pair
            best_mode = best_modes[row_idx, col_idx] 

            total_traj_loss += min_cost[row_idx, col_idx].mean()  # averaged per valid trajectory in cost matrix
            target_confidences[b, row_idx] = best_mode  # Overwrite backgrounds with mode index (as class)

        # Confidence loss
        avg_traj_loss = total_traj_loss / max(1, valid_batches)
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
    B, N, K, T = 1, 4, 32, 10

    pred_trajectory = torch.rand(B, N, K, T, 2)*10
    #              [B,N,K,2]
    pred_trajectory[:, 0, 0, :] = torch.tensor([1.0, 1.0]) + 0.001  # 0->1
    pred_trajectory[:, 1, 1, :] = torch.tensor([1.0, 2.0]) + 0.001  # 1->0
    pred_trajectory[:, 2, 0, :] = torch.tensor([2.0, 1.0]) + 0.001  # 2->2

    pred_confidence = torch.zeros(B, N, K + 1) - torch.ones(K + 1) * 5
    #              [B,N,K]
    pred_confidence[:, 0, 0] *= -1
    pred_confidence[:, 1, 1] *= -1
    pred_confidence[:, 2, 0] *= -1
    pred_confidence[:, 3, K] *= -1  # background
    # pred_confidence[:, 4, K] *= -1  # background

    targets = torch.zeros(B, N, T, 2)
    targets[:, 0, :] = torch.tensor([1.0, 2.0])
    targets[:, 1, :] = torch.tensor([1.0, 1.0])
    targets[:, 2, :] = torch.tensor([2.0, 1.0])
    # targets[:,2,:] = torch.tensor([2.0, 1.0])

    mask = torch.ones(B, N, T)
    mask[:, -1] = 0  # last query not used

    mask[:,0,1] = 0
    # mask[:,1,1] = 0
    # mask[:,2,0] = 0
    # mask[:,2,1] = 0
    # mask[:,2,2] = 0
    # mask[:,2,3] = 0
    # mask[:,2,4] = 0


    criterion = MultiModalHungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    # losses = criterion(outputs, targets, mask)
    losses = criterion(outputs, targets)
    print(losses)
