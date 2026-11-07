"""Focal + Dice loss (Eqs. 1-3). alpha, gamma and eps are common defaults; edit them to match your checkpoint."""
import torch
import torch.nn as nn


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha, self.gamma, self.eps = alpha, gamma, eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        focal = (self.alpha * (1 - pt) ** self.gamma * bce).mean()
        inter = (p * target).sum(dim=(2, 3))
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - ((2 * inter + self.eps) / (den + self.eps)).mean()
        return focal + dice
