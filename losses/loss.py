import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6, include_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        target = target.clamp(0, num_classes - 1)
        pred = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        if not self.include_background:
            dice = dice[:, 1:]
        return 1.0 - dice.mean()


class DeepSupervisionLoss(nn.Module):
    def __init__(self, num_classes: int = 4, main_weight: float = 1.0,
                 aux_weights: Optional[List[float]] = None):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weights = aux_weights if aux_weights is not None else [0.4, 0.3, 0.2]
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, aux_outputs: List[torch.Tensor],
                target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        main_loss = self.dice(logits, target)
        if uncertainty is not None:
            u_mod = 1.0 / (1.0 + uncertainty.mean())
        else:
            u_mod = 1.0
        aux_loss = 0.0
        for aux_logits, weight in zip(aux_outputs, self.aux_weights):
            aux_pred = F.interpolate(aux_logits, size=target.shape[1:],
                                     mode="trilinear", align_corners=False)
            aux_loss += weight * u_mod * self.dice(aux_pred, target)
        return self.main_weight * main_loss + aux_loss
