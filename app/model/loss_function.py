import torch
import torch.nn as nn


class OrdinalLogLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        alpha=1.0,
        reduction="mean",
        distance_matrix=None,
        class_weights=None,
        eps=1e-8,
        ignore_index=-100,
    ):
        super(OrdinalLogLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

        if distance_matrix is not None:
            assert distance_matrix.shape == (num_classes, num_classes), (
                "Distance matrix must be of shape (num_classes, num_classes)"
            )
            self.register_buffer("distance_matrix", distance_matrix.float())
        else:
            idx = torch.arange(num_classes).float()
            default_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            self.register_buffer("distance_matrix", default_matrix)

        if class_weights is not None:
            assert class_weights.shape == (num_classes,), (
                "Class weights must be of shape (num_classes,)"
            )
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, target):
        if logits.numel() == 0:
            return logits.new_tensor(0.0)

        probs = torch.softmax(logits, dim=-1).clamp(max=1 - self.eps)

        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        if not valid_mask.any():
            if self.reduction == "none":
                return logits.new_zeros(target.shape, dtype=logits.dtype)
            return logits.new_tensor(0.0)

        active_probs = probs[valid_mask]
        active_target = target[valid_mask]
        distances = self.distance_matrix[active_target] ** self.alpha
        per_class_loss = -torch.log(1 - active_probs + self.eps)
        loss_active = (per_class_loss * distances).sum(dim=-1)

        if self.class_weights is not None:
            sample_weights = self.class_weights[active_target]
            loss_active = loss_active * sample_weights

        if self.reduction == "none":
            full_loss = logits.new_zeros(target.shape, dtype=logits.dtype)
            full_loss[valid_mask] = loss_active
            return full_loss

        if self.reduction == "mean":
            return loss_active.mean()
        if self.reduction == "sum":
            return loss_active.sum()
        raise ValueError(f"Unsupported reduction: {self.reduction}")
