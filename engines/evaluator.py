import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from utils.metrics import DiceScore, IoUScore, HD95Score


class Evaluator:
    def __init__(self, model: nn.Module, test_loader: DataLoader,
                 config: dict, num_classes: int = 4):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.num_classes = num_classes
        self.device = torch.device(config.get("device", "cuda:0"))
        self.dice_fn = DiceScore(num_classes=num_classes)
        self.iou_fn = IoUScore(num_classes=num_classes)
        self.hd95_fn = HD95Score(num_classes=num_classes)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_dice = []
        all_iou = []
        all_hd95 = []
        for images, targets in self.test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits, _, uncertainty = self.model(images)
            pred = logits.argmax(dim=1)
            all_dice.append(self.dice_fn(pred, targets))
            all_iou.append(self.iou_fn(pred, targets))
            all_hd95.append(self.hd95_fn(pred, targets))
        avg_dice = torch.stack(all_dice).mean(dim=0)
        avg_iou = torch.stack(all_iou).mean(dim=0)
        avg_hd95 = torch.tensor(all_hd95).mean(dim=0) if len(all_hd95) > 0 else torch.zeros(self.num_classes - 1)
        results = {
            "dice_wt": avg_dice[1].item(),
            "dice_tc": avg_dice[2].item() if self.num_classes > 2 else 0.0,
            "dice_et": avg_dice[3].item() if self.num_classes > 3 else 0.0,
            "mean_dice": avg_dice[1:].mean().item(),
            "mean_iou": avg_iou[1:].mean().item(),
            "mean_hd95": avg_hd95.mean().item(),
        }
        return results

    @torch.no_grad()
    def predict_with_uncertainty(self, images: torch.Tensor):
        self.model.eval()
        images = images.to(self.device)
        logits, _, uncertainty = self.model(images)
        pred = logits.argmax(dim=1)
        return pred.cpu(), uncertainty.cpu() if uncertainty is not None else None
