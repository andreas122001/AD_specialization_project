import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional
from torch import nn

class MultiModalHungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.trajectory_loss = nn.L1Loss(reduction="mean")
        self.confidence_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, mask=None):
        pred_trajectories, pred_confidences = outputs  # [B, N, K, F, 2], [B, N, K+1]
        batch_size, n_queries, n_modes, F, _ = pred_trajectories.shape

        # Validity check (same as before)
        valid_indices = targets.abs().sum(dim=(2, 3)).ne(0).long()  # [B, N]
        target_confidences = torch.zeros(batch_size, n_queries).long()  # [B, N]

        total_traj_loss = 0.0

        for b in range(batch_size):
            pred_traj = pred_trajectories[b]  # [N, K, F, 2]
            targets_b = targets[b]  # [N, F, 2]
            valid_mask = valid_indices[b].bool()  # [N]
            valid_targets = targets_b[valid_mask]  # [n_valid, F, 2]
            n_valid = valid_mask.sum()

            if n_valid == 0:
                continue
            
            # Compute the multimodal cost matrix
            cost_matrix = torch.zeros(
                (n_queries, n_valid), device=pred_trajectory.device
            )
            # Get the best selected mode for each query-target pair
            selected_modes = torch.zeros(
                n_queries, n_valid, device=pred_trajectory.device
            ).long()
            for i in range(n_queries):
                for j in range(n_valid):
                    min_cost = float("inf")
                    best_mode = -1
                    for k in range(n_modes):
                        cost = self.trajectory_loss(
                            pred_traj[i, k], valid_targets[j]
                        )
                        if cost < min_cost:
                            min_cost = cost
                            best_mode = k
                    # Store the min cost and mode of min cost
                    cost_matrix[i, j] = min_cost
                    selected_modes[i, j] = best_mode + 1  # 0 is reserved for no match

            print(cost_matrix)

            # Hungarian matching on minimal cost modes
            with torch.no_grad():
                cost_np = cost_matrix.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)

            # Assign the best mode classes to the target confidences
            target_confidences[b, row_idx] = selected_modes[row_idx, col_idx]

            matched_modes = selected_modes[row_idx, col_idx]  # [n_valid]
            
            matched_preds = pred_traj[row_idx, matched_modes - 1]  # [n_valid, F, 2]
            matched_targets = valid_targets[col_idx]  # [n_valid, F, 2]

            # Need to mask the matched preds as well
            # This is done per-batch as we need the correct ordering
            if mask is not None:
                original_indices = torch.where(valid_mask)[0]
                mask_index = original_indices[col_idx]
                matched_preds = matched_preds * mask[b][mask_index].unsqueeze(2)

            total_traj_loss += self.trajectory_loss(matched_preds, matched_targets)
    
        # Confidence loss (sigmoid + BCE)
        avg_traj_loss = total_traj_loss / batch_size

        pred_confidences = pred_confidences.view(-1, n_modes + 1)  # [B*N, K + 1]
        target_confidences = target_confidences.view(-1).long()  # [B*N]
        avg_conf_loss = self.confidence_loss(pred_confidences, target_confidences)

        return {
            "loss_trajectories": avg_traj_loss,
            "loss_trajectory_confidence": avg_conf_loss
        }

if __name__ == "__main__":
    B, N, K, F = 1, 3, 1, 2

    pred_trajectory = torch.zeros(B, N, K, F, 2)
    #              [B,N,K,2]
    pred_trajectory[:,1,0,:] = torch.tensor([1.0, 2.0]) + 0.001
    pred_trajectory[:,0,0,:] = torch.tensor([1.0, 1.0]) + 0.001
    pred_trajectory[:,2,0,:] = torch.tensor([2.0, 1.0]) + 0.001


    pred_confidence = torch.zeros(B, N, K+1) - torch.ones(K+1)*10
    #              [B,N,K]
    pred_confidence[:,1,1] *= -1
    pred_confidence[:,0,1] *= -1
    pred_confidence[:,2,1] *= -1

    print(pred_confidence)

    targets = torch.zeros(B, N, F, 2)
    targets[:,0,:] = torch.tensor([1.0, 2.0])
    targets[:,1,:] = torch.tensor([1.0, 1.0])
    targets[:,2,:] = torch.tensor([2.0, 1.0])
    # targets[:,2,:] = torch.tensor([2.0, 1.0])
    
    mask = torch.ones(B, N, F)
    # mask[:,0,3] = 0

    criterion = MultiModalHungarianLoss()
    outputs = (pred_trajectory, pred_confidence)
    losses = criterion(outputs, targets, mask)
    print(losses)
