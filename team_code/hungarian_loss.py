import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional


class HungarianLoss(torch.nn.Module):
    def __init__(self):
        """
        Args:
            num_queries: Number of queries N (e.g., 10)
        """
        super().__init__()
        self.trajectory_loss = torch.nn.L1Loss(reduction="mean")
        self.confidence_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[torch.Tensor, torch.Tensor]:
        """
        Compute the Hungarian loss for predicted 2D trajectories with batch B, number of objects N, for future timesteps F.

        Args:
            outputs: tuple containing predicted trajectories and confidence scores.
                    - pred_trajectory: Tensor of shape [B, N, F, 2]
                    - pred_confidence: Tensor of shape [B, N, 2] (logits for [padded, valid] classes)
            targets: padded ground truth trajectories [B, N, F, 2].
                    Valid trajectories have non-zero values, while padded ones are zeros.
            mask: trajectory mask for the GT trajectories [B, N, F].
                    Useful for ignoring parts of GT trajectories if some points are missing.
        """
        pred_trajectory, pred_confidence = outputs  # pred_trajectory: [B, N, F, 2]
        batch_size, num_queries, F, _ = pred_trajectory.shape  # pred_confidence: [B, N, 2]

        # Compute a per-trajectory validity flag for ground truth.
        # Here, if the sum of absolute values in a trajectory is not zero, it is considered valid.
        # Otherwise, it is most certainly just padding.
        valid_indices = targets.abs().sum(dim=(2, 3)).not_equal(0).long()  # Shape: [B, N]

        # Mask out invalid trajectory points
        # if some points are missing (e.g. actor is missing for some frame), we exclude them
        # We don't want to penalize the model for bad GTs
        if mask is not None:
            targets = targets * mask.unsqueeze(3)

        # Initialize the target confidences, which will contain the reordered valid targets (after matching)
        target_confidences = torch.zeros(batch_size, num_queries).long().to(pred_confidence.device)

        total_traj_loss = 0.0

        # For each batch element, compute the optimal matching.
        for b in range(batch_size):

            # For the current batch element: predicted trajectories [N, F, 2]
            preds_traj = pred_trajectory[b]  # Shape: [N, F, 2]
            targets_b = targets[b]  # Shape: [N, F, 2]

            # Mask out invalid targets, and count number of valids
            valid_mask = valid_indices[b].bool()  # Shape: [N]
            valid_targets = targets_b[valid_mask]  # Shape: [num_valid, F, 2]
            num_valid = valid_mask.long().sum()

            # If no valid targets exist, continue. 
            # We don't need to match it since they are all zero anyway.
            if num_valid == 0:
                continue

            # Build cost matrix [num_queries, num_valid] using the L1 loss.
            cost_matrix = torch.zeros(
                (num_queries, num_valid), device=pred_trajectory.device
            )
            for i in range(num_queries):
                for j in range(num_valid):
                    # Compute cost for each pred/target-pair.
                    cost_matrix[i, j] = self.trajectory_loss(preds_traj[i], valid_targets[j])

            # Hungarian algorithm is non-differentiable, so we detach
            with torch.no_grad():
                # Get minimum loss indices.
                cost_np = cost_matrix.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Use the matching indices to gather predictions and corresponding targets
            matched_preds = preds_traj[row_idx]  # [num_valid, F, 2]
            matched_targets = valid_targets[col_idx]  # [num_valid, F, 2]

            # Need to mask the matched preds as well
            # This is done per-batch as we need the correct ordering
            if mask is not None:
                original_indices = torch.where(valid_mask)[0]
                mask_index = original_indices[col_idx]
                matched_preds = matched_preds * mask[b][mask_index].unsqueeze(2)

            # Update the target confidences
            target_confidences[b, row_idx] = 1  # matched indices should be 1

            # Calculate the trajectory loss for the matched pairs
            loss_traj = self.trajectory_loss(matched_preds, matched_targets)
            total_traj_loss += loss_traj

        # Average the trajectory loss over the batch.
        avg_traj_loss = total_traj_loss / batch_size

        # Compute the confidence loss.
        # Here, we assume that pred_confidence has shape [B, N, 2] (logits for two classes: padded vs valid)
        # and target_valid is of shape [B, N] with values 0 (padded) or 1 (valid).
        # Reshape to combine batch and queries.
        conf_logits = pred_confidence.view(-1, 2)  # [B*N, 2]
        target_confidences = target_confidences.view(-1).long()  # [B*N]
        avg_conf_loss = self.confidence_loss(conf_logits, target_confidences)

        losses = {
            "loss_trajectories": avg_traj_loss,
            "loss_trajectory_confidence": avg_conf_loss,
        }
        return losses


if __name__ == "__main__":
    # Example usage
    B, N, F = 1, 3, 4
    pred_trajectory = torch.zeros(B, N, F, 2)
    pred_trajectory[:,2,:] = torch.tensor([1.0, 1.0])
    pred_trajectory[:,0,:] = torch.tensor([2.1, 1.0])

    pred_confidence = torch.zeros(B, N, 2) + torch.tensor([1.0, -1.0])*10
    pred_confidence[:,0,:] = torch.tensor([-1.0, 1.0])*10
    pred_confidence[:,2,:] = torch.tensor([-1.0, 1.0])*10

    targets = torch.zeros(B, N, F, 2)
    targets[:,0,:] = torch.tensor([1.0, 1.0])
    targets[:,2,:] = torch.tensor([2.0, 1.0])
    
    mask = torch.ones(B, N, F)
    mask[:,0,3] = 0

    criterion = HungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    losses = criterion(outputs, targets, mask)
    print(losses)