import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional
from torch import nn

class MultiModalHungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.trajectory_loss = nn.L1Loss(reduction="none")
        self.confidence_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, mask=None):
        pred_trajectories, pred_confidences = outputs  # [B, N, K, F, 2], [B, N, K]
        batch_size, n_queries, n_modes, F, _ = pred_trajectories.shape

        # Validity check (same as before)
        valid_indices = targets.abs().sum(dim=(2, 3)).ne(0).long()  # [B, N]
        target_confidences = torch.zeros(batch_size, n_queries, n_modes).long()  # [B, N, K]

        total_traj_loss = 0.0

        for b in range(batch_size):
            pred_traj = pred_trajectories[b]  # [N, K, F, 2]
            targets_b = targets[b]  # [N, F, 2]
            valid_mask = valid_indices[b].bool()  # [N]
            valid_targets = targets_b[valid_mask]  # [num_valid, F, 2]
            num_valid = valid_mask.sum()

            if num_valid == 0:
                continue

            # Compute cost matrix [N, K, num_valid]
            diff = torch.abs(
                pred_traj.unsqueeze(2) - valid_targets.unsqueeze(0).unsqueeze(0)
            )  # [N, K, num_valid, F, 2]
            cost_matrix = diff.mean(dim=(-1, -2))  # [N, K, num_valid]

            # Find minimal cost per query-target pair and best mode
            min_cost_per_query, best_mode = cost_matrix.min(dim=1)  # [N, num_valid], [N, num_valid]

            # Hungarian matching on minimal costs
            with torch.no_grad():
                cost_np = min_cost_per_query.cpu().detach().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Assign confidences and compute loss
            for i, j in zip(row_idx, col_idx):
                best_k = best_mode[i, j].item()
                target_confidences[b, i, best_k] = 1

                # Trajectory loss (mask if needed)
                pred_traj_masked = pred_traj[i, best_k]
                if mask is not None:
                    pred_traj_masked = pred_traj_masked * mask[b, i].unsqueeze(-1)

                total_traj_loss += self.trajectory_loss(pred_traj_masked, valid_targets[j]).mean()
    
        # Confidence loss (sigmoid + BCE)
        avg_traj_loss = total_traj_loss / batch_size

        pred_confidences = pred_confidences.view(-1, 2)  # [B*N, 2]
        target_confidences = target_confidences.view(-1).long()  # [B*N]
        avg_conf_loss = self.confidence_loss(pred_confidences, target_confidences)

        return {
            "loss_trajectories": avg_traj_loss,
            "loss_trajectory_confidence": avg_conf_loss
        }

if __name__ == "__main__":
    B, N, K, F = 1, 3, 5, 3

    pred_trajectory = torch.zeros(B, N, K, F, 2)
    pred_trajectory[:,0,1,:] = torch.tensor([1.0, 1.0])
    # pred_trajectory[:,0,:,:] = torch.tensor([2.1, 1.0])

    pred_confidence = torch.zeros(B, N, K, 2) + torch.tensor([1.0, -1.0])*10
    pred_confidence[:,0,0] = torch.tensor([-1.0, 1.0])*10
    # pred_confidence[:,2,:] = torch.tensor([-1.0, 1.0])*10

    targets = torch.zeros(B, N, F, 2)
    targets[:,0,:] = torch.tensor([1.0, 1.0])
    # targets[:,2,:] = torch.tensor([2.0, 1.0])
    
    mask = torch.ones(B, N, F)
    # mask[:,0,3] = 0

    criterion = MultiModalHungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    losses = criterion(outputs, targets, mask)
    print(losses)
