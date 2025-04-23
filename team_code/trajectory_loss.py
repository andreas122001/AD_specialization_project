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

    def forward(self, outputs, targets, mask=None):
        pred_trajectories, pred_confidences = outputs  # [B, N, K, F, 2], [B, N, K+1]
        batch_size, n_queries, n_modes, F, _ = pred_trajectories.shape
        
        # Validity check (vectorized)
        valid_mask = targets.abs().sum(dim=(2, 3)).ne(0)  # [B, N]
        target_confidences = torch.zeros(
            batch_size, n_queries, device=pred_trajectories.device
        ).long()

        total_traj_loss = 0.0

        for b in range(batch_size):
            pred_traj = pred_trajectories[b]  # [N, K, F, 2]
            targets_b = targets[b]  # [N, F, 2]
            valid_mask_b = valid_mask[b]  # [N]
            valid_targets = targets_b[valid_mask_b]  # [n_valid, F, 2]
            n_valid = valid_targets.shape[0]

            if n_valid == 0:
                continue

            # Vectorized cost computation: [N, K, n_valid]
            # Expand dimensions for broadcasting
            pred_exp = pred_traj.unsqueeze(2).expand(-1, -1, n_valid, -1, -1)
            tgt_exp = valid_targets.unsqueeze(0).unsqueeze(0).expand(pred_traj.size(0), n_modes, -1, -1, -1)
            
            cost_matrix = self.trajectory_loss(pred_exp, tgt_exp).mean(dim=(3, 4))  # [N, K, n_valid]

            # Find best mode per query-target pair: [N, n_valid]
            min_cost, best_modes = cost_matrix.min(dim=1)  # [N, n_valid]

            # Hungarian matching on minimal cost modes
            with torch.no_grad():
                cost_np = min_cost.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Assign confidences
            target_confidences[b, row_idx] = best_modes[row_idx, col_idx] + 1

            # Compute trajectory loss for matched pairs
            matched_preds = pred_traj[row_idx, best_modes[row_idx, col_idx]]  # [n_valid, F, 2]
            matched_targets = valid_targets[col_idx]  # [n_valid, F, 2]

            if mask is not None:
                original_indices = torch.where(valid_mask_b)[0]
                mask_index = original_indices[col_idx]
                matched_preds = matched_preds * mask[b][mask_index].unsqueeze(-1)

            total_traj_loss += self.trajectory_loss(matched_preds, matched_targets).mean()

        # Confidence loss
        avg_traj_loss = total_traj_loss / batch_size
        pred_confidences = pred_confidences.view(-1, n_modes + 1)
        target_confidences = target_confidences.view(-1)
        avg_conf_loss = self.confidence_loss(pred_confidences, target_confidences)

        return {
            "loss_trajectories": avg_traj_loss,
            "loss_trajectory_confidence": avg_conf_loss,
        }


if __name__ == "__main__":
    B, N, K, F = 2, 3, 1, 2

    pred_trajectory = torch.zeros(B, N, K, F, 2)
    #              [B,N,K,2]
    pred_trajectory[:, 1, 0, :] = torch.tensor([1.0, 2.0]) + 0.001
    pred_trajectory[:, 0, 0, :] = torch.tensor([1.0, 1.0]) + 0.001
    pred_trajectory[:, 2, 0, :] = torch.tensor([2.0, 1.0]) + 0.001

    pred_confidence = torch.zeros(B, N, K + 1) - torch.ones(K + 1) * 10
    #              [B,N,K]
    pred_confidence[:, 1, 1] *= -1
    pred_confidence[:, 0, 1] *= -1
    pred_confidence[:, 2, 1] *= -1

    targets = torch.zeros(B, N, F, 2)
    targets[:, 0, :] = torch.tensor([1.0, 2.0])
    targets[:, 1, :] = torch.tensor([1.0, 1.0])
    targets[:, 2, :] = torch.tensor([2.0, 1.0])
    # targets[:,2,:] = torch.tensor([2.0, 1.0])

    mask = torch.ones(B, N, F)
    # mask[:,0,3] = 0

    criterion = MultiModalHungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    losses = criterion(outputs, targets, mask)
    print(losses)
