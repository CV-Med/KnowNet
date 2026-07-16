import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4, smooth: float = 1e-6):
    target = target.clamp(0, num_classes - 1)
    pred_one_hot = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()
    target_one_hot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()
    intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
    union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class DiceScore:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        dice = dice_per_class(pred, target, self.num_classes)
        return dice.mean(dim=0)


def region_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    target = target.clamp(0, 3)
    regions = {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    }
    results = {}
    for name, labels in regions.items():
        p = torch.isin(pred, torch.tensor(labels, device=pred.device)).float()
        t = torch.isin(target, torch.tensor(labels, device=target.device)).float()
        inter = (p * t).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        results[name] = ((2.0 * inter + smooth) / (union + smooth)).mean().item()
    return results

class IoUScore:
    def __init__(self, num_classes: int = 4, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        target = target.clamp(0, self.num_classes - 1)
        pred_one_hot = F.one_hot(pred, self.num_classes).permute(0, 4, 1, 2, 3).float()
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3, 4))
        union = pred_one_hot.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.mean(dim=0)

class HD95Score:
    def __init__(self, num_classes: int = 4, voxel_spacing: tuple = (1.0, 1.0, 1.0)):
        self.num_classes = num_classes
        self.voxel_spacing = voxel_spacing

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        hd95_list = []
        for c in range(1, self.num_classes):
            pred_c = (pred_np == c).astype(np.float32)
            target_c = (target_np == c).astype(np.float32)
            if pred_c.sum() == 0 or target_c.sum() == 0:
                hd95_list.append(0.0)
                continue
            from scipy.ndimage import distance_transform_edt
            pred_dt = distance_transform_edt(1 - pred_c, sampling=self.voxel_spacing)
            target_dt = distance_transform_edt(1 - target_c, sampling=self.voxel_spacing)
            distances = np.concatenate([pred_dt[target_c > 0], target_dt[pred_c > 0]])
            hd95 = np.percentile(distances, 95)
            hd95_list.append(hd95)
        return np.array(hd95_list)
