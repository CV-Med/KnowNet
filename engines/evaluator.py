"""Evaluation engine for KnowNet.

Segmentation is evaluated on the complete volume with sliding-window inference;
metrics are aggregated per case (mean +/- std over cases) and then averaged over
the split.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import resolve_device
from utils.inference import center_crop, sliding_window_logits
from utils.metrics import DiceScore, HD95Score, IoUScore, region_dice


class Evaluator:
    def __init__(self, model: nn.Module, data_loader: DataLoader, config: dict,
                 device: Optional[str] = None, sliding_window: Optional[bool] = None,
                 compute_hd95: Optional[bool] = None):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = resolve_device(device or config.get("device", "auto"))
        self.model.to(self.device)
        self.model.eval()

        data_config = config.get("data", {})
        evaluation_config = config.get("evaluation", {})
        self.num_classes = data_config.get("num_classes", 4)
        self.crop_size = tuple(data_config.get("crop_size", (128, 128, 128)))
        self.window_size = tuple(evaluation_config.get("window_size", self.crop_size))
        self.stride = tuple(evaluation_config.get("stride", (64, 64, 64)))
        self.sliding_window = (
            evaluation_config.get("sliding_window", True)
            if sliding_window is None else sliding_window
        )
        self.compute_hd95 = (
            evaluation_config.get("compute_hd95", True)
            if compute_hd95 is None else compute_hd95
        )

        self.dice_fn = DiceScore(num_classes=self.num_classes)
        self.iou_fn = IoUScore(num_classes=self.num_classes)
        self.hd95_fn = HD95Score(num_classes=self.num_classes)

    @torch.no_grad()
    def _logits(self, image: torch.Tensor) -> torch.Tensor:
        if self.sliding_window and tuple(image.shape[2:]) != self.crop_size:
            return sliding_window_logits(
                self.model, image, window_size=self.window_size, stride=self.stride
            )
        return self.model(center_crop(image, self.crop_size))[0]

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Return case-averaged metrics (all Dice/IoU values in [0, 1])."""
        per_case = []
        for images, targets in self.data_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            for i in range(images.shape[0]):
                logits = self._logits(images[i:i + 1])
                pred = logits.argmax(dim=1)
                target = targets[i:i + 1]

                dice = self.dice_fn(pred, target).cpu().numpy()
                iou = self.iou_fn(pred, target).cpu().numpy()
                regions = region_dice(pred, target)
                case = {
                    "dice_mean": float(dice.mean()),
                    "iou_mean": float(iou.mean()),
                    "dice_WT": regions["WT"],
                    "dice_TC": regions["TC"],
                    "dice_ET": regions["ET"],
                }
                if self.compute_hd95:
                    hd95 = self.hd95_fn(pred, target)
                    case["hd95_mean"] = float(np.mean(hd95))
                    case["hd95_WT"] = float(hd95[0])
                    case["hd95_TC"] = float(hd95[1])
                    case["hd95_ET"] = float(hd95[2])
                per_case.append(case)

        if not per_case:
            raise RuntimeError("Evaluation loader produced no cases")

        results = {}
        keys = per_case[0].keys()
        for key in keys:
            values = np.array([case[key] for case in per_case], dtype=np.float64)
            results[key] = float(values.mean())
            results[f"{key}_std"] = float(values.std())
        results["num_cases"] = len(per_case)
        results["per_case"] = per_case

        # Aliases kept for compatibility with existing reporting code.
        results["mean_dice"] = results["dice_mean"]
        results["mean_iou"] = results["iou_mean"]
        results["dice_wt"] = results["dice_WT"]
        results["dice_tc"] = results["dice_TC"]
        results["dice_et"] = results["dice_ET"]
        if self.compute_hd95:
            results["mean_hd95"] = results["hd95_mean"]
        return results

    def summary(self, results: Optional[Dict[str, float]] = None) -> str:
        results = results or self.evaluate()
        lines = [
            f"Cases: {results['num_cases']}",
            f"Dice  WT {results['dice_WT'] * 100:.2f} | "
            f"TC {results['dice_TC'] * 100:.2f} | "
            f"ET {results['dice_ET'] * 100:.2f} | "
            f"Mean {results['dice_mean'] * 100:.2f} "
            f"(+/- {results['dice_mean_std'] * 100:.2f})",
        ]
        if "hd95_mean" in results:
            lines.append(
                f"HD95  WT {results['hd95_WT']:.2f} | "
                f"TC {results['hd95_TC']:.2f} | "
                f"ET {results['hd95_ET']:.2f} | "
                f"Mean {results['hd95_mean']:.2f}"
            )
        return "\n".join(lines)
