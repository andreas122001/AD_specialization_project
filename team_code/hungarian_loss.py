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
        Compute the Hungarian loss for predicted 2D trajectories with batch B, number of trajectories N, for future timesteps F.

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
        batch_size, num_queries, F, _ = pred_trajectory.shape

        # Compute a per-trajectory validity flag for ground truth.
        # Here, if the sum of absolute values in a trajectory is greater than zero, it is considered valid.
        target_valid = (targets.abs().sum(dim=(2, 3)) > 0).long()  # Shape: [B, N]

        total_traj_loss = 0.0

        # For each batch element, compute the optimal matching.
        for b in range(batch_size):
            # For the current batch element: predicted trajectories [N, F, 2]
            preds = pred_trajectory[b]  # Shape: [N, F, 2]
            targets_b = targets[b]  # Shape: [N, F, 2]

            # Mask for valid trajectory points
            # if some points are missing (e.g. actor went outside bounds), we exclude them
            # We don't want to penalize the model for bad GTs
            if mask is not None:
                preds = preds * mask[b].unsqueeze(2)
                targets_b = targets_b * mask[b].unsqueeze(2)

            # Mask for number of actors
            valid_mask = target_valid[b].bool()  # Shape: [N]

            # Extract only the valid ground truth trajectories.
            valid_targets = targets_b[valid_mask]  # Shape: [num_valid, F, 2]
            num_valid = valid_targets.shape[0]

            # If no valid targets exist, continue.
            if num_valid == 0:
                continue

            # Build the cost matrix [num_queries, num_valid] using the L1 loss.
            cost_matrix = torch.zeros(
                (num_queries, num_valid), device=pred_trajectory.device
            )
            for i in range(num_queries):
                for j in range(num_valid):
                    # Compute cost for each pair.
                    cost_matrix[i, j] = self.trajectory_loss(preds[i], valid_targets[j])

            # Hungarian algorithm is non-differentiable, so we detach
            with torch.no_grad():
                # Convert cost matrix to numpy.
                cost_np = cost_matrix.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)

            # Use the matching indices to gather predictions and corresponding targets.
            matched_preds = preds[row_ind]  # [num_valid, F, 2]
            matched_targets = valid_targets[col_ind]  # [num_valid, F, 2]

            # Compute the trajectory loss for the matched pairs.
            loss_traj = self.trajectory_loss(matched_preds, matched_targets)
            total_traj_loss += loss_traj

        # Average the trajectory loss over the batch.
        avg_traj_loss = total_traj_loss / batch_size

        # Compute the confidence loss.
        # Here, we assume that pred_confidence has shape [B, N, 2] (logits for two classes: padded vs valid)
        # and target_valid is of shape [B, N] with values 0 (padded) or 1 (valid).
        # Reshape to combine batch and queries.
        conf_logits = pred_confidence.view(-1, 2)  # [B*N, 2]
        conf_targets = target_valid.view(-1)  # [B*N]
        loss_conf = self.confidence_loss(conf_logits, conf_targets)

        losses = {
            "loss_traj": avg_traj_loss,
            "loss_traj_confidence": loss_conf,
        }
        return losses
