import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = RobustCrossEntropyLoss()

    def forward(self, inputs, targets):
        CE_loss = self.ce(inputs, targets)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-CE_loss)
        F_loss = at * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

if __name__ == '__main__':
    y_true = torch.randint(0, 1, (2, 1, 12, 384, 384), dtype=torch.float32)  # 示例掩码图1
    y_pred = torch.rand(2, 2, 12, 384, 384, dtype=torch.float32)  # 示例掩码图2
    
    loss_fn = FocalLoss(gamma=2., alpha=.25)
    loss = loss_fn(y_pred, y_true)
    print("Focal Loss:", loss.item())