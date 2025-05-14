# src/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal-Loss (α, γ) поверх logits.
    alpha  – float или [C]  (баланс классов)
    gamma  – float (степень «фокусировки»)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        if isinstance(alpha, (float, int)):
            # binary-case: [1-α, α]
            self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)  # list/np
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits  : [B, C]   (сырые выходы модели)
        targets : [B]      (LongTensor меток)
        """
        logpt = F.log_softmax(logits, dim=1)        # log(p_t)
        pt    = torch.exp(logpt)                    # p_t

        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt    = pt.gather( 1, targets.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha.to(logits.device)[targets]
        loss = - alpha_t * (1 - pt) ** self.gamma * logpt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss                  # 'none'
