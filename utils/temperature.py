"""Post-training EvCal temperature fitting.

The temperature is *not* learned by gradient descent: after training, tumour
voxels of the validation split are collected, the Brier score is evaluated on a
500-point logarithmic grid of ``tau`` and the minimiser is stored as the EvCal
buffer of the checkpoint.  ``tau`` only rescales the evidence concentration, so
the argmax, and therefore Dice/HD95, is unchanged.
"""

import json
import os
from typing import Dict, Optional, Tuple

import torch

from utils.inference import center_crop, sliding_window_logits


@torch.no_grad()
def collect_temperature_data(config: dict, model, loader, device: torch.device,
                             max_voxels: Optional[int] = None,
                             max_voxels_per_case: Optional[int] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect a subsample of tumour-voxel logits / labels from ``loader``."""
    calibration_config = config.get("calibration", {})
    max_voxels = int(max_voxels or calibration_config.get("max_voxels", 100_000))
    max_per_case = int(
        max_voxels_per_case or calibration_config.get("max_voxels_per_case", 2_048)
    )
    evaluation_config = config.get("evaluation", {})
    crop_size = tuple(config["data"].get("crop_size", (128, 128, 128)))
    window_size = tuple(evaluation_config.get("window_size", crop_size))
    stride = tuple(evaluation_config.get("stride", (64, 64, 64)))
    sliding = bool(evaluation_config.get("sliding_window", True))

    logits_chunks, label_chunks = [], []
    total = 0
    model.eval()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        for i in range(images.shape[0]):
            image = images[i:i + 1]
            if sliding and tuple(image.shape[2:]) != crop_size:
                logits = sliding_window_logits(model, image, window_size, stride)
            else:
                logits = model(center_crop(image, crop_size))[0]
            target = targets[i]
            mask = target > 0
            coordinates = mask.nonzero(as_tuple=False)
            if coordinates.shape[0] == 0:
                continue
            if coordinates.shape[0] > max_per_case:
                choice = torch.randperm(coordinates.shape[0], device=device)[:max_per_case]
                coordinates = coordinates[choice]
            chunks = logits[0, :, coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
            logits_chunks.append(chunks.transpose(0, 1).cpu())
            label_chunks.append(
                target[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]].cpu()
            )
            total += chunks.shape[1]
            if total >= max_voxels:
                break
        if total >= max_voxels:
            break

    if not logits_chunks:
        raise RuntimeError("No tumour voxels found in the validation split; cannot fit tau.")

    logits = torch.cat(logits_chunks, dim=0)        # (M, C)
    labels = torch.cat(label_chunks, dim=0)         # (M,)
    # brier_score expects (B, C, 1, 1, M) probabilities and (B, 1, 1, M) labels.
    logits = logits.transpose(0, 1).view(1, logits.shape[1], 1, 1, -1)
    labels = labels.view(1, 1, 1, -1)
    return logits, labels


def calibrate_model(config: dict, model, val_loader, device: torch.device,
                    checkpoint_path: str,
                    output_json: str = "calibration.json") -> Dict[str, float]:
    """Fit the EvCal temperature on ``val_loader`` and store it in a checkpoint."""
    if getattr(model, "evcal", None) is None:
        raise RuntimeError("EvCal is disabled in this model; nothing to calibrate.")

    calibration_config = config.get("calibration", {})
    logits, labels = collect_temperature_data(config, model, val_loader, device)
    tau_star, info = model.evcal.fit_temperature(
        logits, labels,
        tau_min=float(calibration_config.get("tau_min", 0.05)),
        tau_max=float(calibration_config.get("tau_max", 5.0)),
        num=int(calibration_config.get("tau_grid_points", 500)),
    )
    print(
        f"[calibrate] tau*={tau_star:.4f} "
        f"(Brier={info['best_brier']:.4f}, tumour voxels={logits.shape[-1]})"
    )

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "tau": tau_star,
        "calibration": {"tau": tau_star, "brier": info["best_brier"]},
    }, checkpoint_path)
    with open(output_json, "w") as f:
        json.dump({"tau_star": tau_star, "brier": info["best_brier"], "grid": info},
                  f, indent=2)
    return {"tau": tau_star, "brier": info["best_brier"]}
