import torch
import torch.nn as nn


class OrdinalLogLoss(nn.Module):
    def __init__(
            self,
            num_classes,
            alpha=1.0,
            reduction='mean',
            distance_matrix=None,
            class_weights=None,
            eps=1e-8
    ):
        super(OrdinalLogLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

        if distance_matrix is not None:
            assert distance_matrix.shape == (num_classes, num_classes), \
                "Distance matrix must be of shape (num_classes, num_classes)"
            self.register_buffer('distance_matrix', distance_matrix.float())
        else:
            idx = torch.arange(num_classes).float()
            default_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
            self.register_buffer('distance_matrix', default_matrix)

        if class_weights is not None:
            assert class_weights.shape == (num_classes,), \
                "Class weights must be of shape (num_classes,)"
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1).clamp(max=1 - self.eps)
        distances = self.distance_matrix[target] ** self.alpha
        per_class_loss = -torch.log(1 - probs + self.eps)
        loss = (per_class_loss * distances).sum(dim=1)  # shape (batch_size,)
        
        # Apply class weights
        if self.class_weights is not None:
            sample_weights = self.class_weights[target]
            loss = loss * sample_weights

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
