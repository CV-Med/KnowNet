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
    """Dice loss with deep supervision and the EvCal auxiliary weight.

    Deep supervision with an uncertainty-derived auxiliary weight::

        L_aux   = sum_l lambda_l * sg[w_U] * L_Dice(y_hat_l, y)
        L_total = L_Dice(y_hat, y) + L_aux
        w_U     = 1 / (1 + mean(U))          # stop-gradient

    ``sg[.]`` is a stop-gradient: the uncertainty weight scales the auxiliary
    terms but no uncertainty gradient flows back into the encoder through it,
    so the evidential head is optimised by the main Dice term only while its
    uncertainty still modulates the auxiliary supervision.

    ``aux_weight_mode``:
      * ``"uncertainty"`` (default) -- detached ``w_U``.
      * ``"constant"``            -- fixed ``constant_aux_weight`` for the
                                     constant control (w_U = 0.75).
      * ``"none"``                -- no auxiliary weighting.
    """

    def __init__(self, num_classes: int = 4, main_weight: float = 1.0,
                 aux_weights: Optional[List[float]] = None,
                 aux_weight_mode: str = "uncertainty",
                 constant_aux_weight: float = 0.75):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weights = aux_weights if aux_weights is not None else [0.4, 0.3, 0.2, 0.1]
        self.aux_weight_mode = aux_weight_mode
        self.constant_aux_weight = constant_aux_weight
        if aux_weight_mode not in ("uncertainty", "constant", "none"):
            raise ValueError(
                f"Unknown aux_weight_mode '{aux_weight_mode}', expected "
                f"'uncertainty', 'constant' or 'none'."
            )
        self.dice = DiceLoss(num_classes=num_classes)

    def _aux_weight(self, uncertainty: Optional[torch.Tensor]):
        if self.aux_weight_mode == "uncertainty" and uncertainty is not None:
            # Detached stop-gradient weight: no gradient into the evidential head.
            return (1.0 / (1.0 + uncertainty.mean())).detach()
        if self.aux_weight_mode == "constant":
            return self.constant_aux_weight
        return 1.0

    def forward(self, logits: torch.Tensor, aux_outputs: List[torch.Tensor],
                target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        main_loss = self.dice(logits, target)
        u_weight = self._aux_weight(uncertainty)
        aux_loss = 0.0
        for aux_logits, weight in zip(aux_outputs, self.aux_weights):
            aux_pred = F.interpolate(aux_logits, size=target.shape[1:],
                                     mode="trilinear", align_corners=False)
            aux_loss = aux_loss + weight * u_weight * self.dice(aux_pred, target)
        return self.main_weight * main_loss + aux_loss
