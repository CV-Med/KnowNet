"""Batch inference for KnowNet with whole-volume sliding-window evaluation.

For every case under ``--input-dir`` (default ``predict/``) the four MRI
modalities are z-score normalised per scan and per channel, the complete volume
is segmented with sliding-window inference (128^3 window, stride 64), and the
following artefacts are written to ``--output-dir``:

* ``{case_id}_pred.nii.gz``: predicted label map (0/1/2/3)
* ``{case_id}_overlay.png``: FLAIR slice with GT and prediction contours
* ``{case_id}_uncertainty.png``: FLAIR slice with the EvCal uncertainty map
* ``samples_info.json``: Dice / HD95 and mean uncertainty per case

The bundled sample cases are stored with Git LFS; run ``git lfs pull`` (or point
``--input-dir`` at a local BraTS folder) before predicting.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.device import resolve_device, device_summary
from utils.inference import sliding_window_logits
from utils.metrics import HD95Score, region_dice

MODALITIES = ["flair", "t1", "t1ce", "t2"]
CLASS_NAMES = ["background", "necrotic core", "edema", "enhancing tumor"]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def load_volume(case_dir: str, case_id: str) -> Optional[np.ndarray]:
    volumes = []
    for modality in MODALITIES:
        path = os.path.join(case_dir, f"{case_id}_{modality}.nii.gz")
        if not os.path.exists(path):
            print(f"[skip] {case_id}: missing {os.path.basename(path)}")
            return None
        if is_lfs_pointer(path):
            print(
                f"[skip] {case_id}: {os.path.basename(path)} is a Git LFS pointer. "
                "Run `git lfs pull` or pass --input-dir pointing to the raw BraTS data."
            )
            return None
        volumes.append(nib.load(path).get_fdata(dtype=np.float32))
    return np.stack(volumes, axis=0)


def load_seg(case_dir: str, case_id: str) -> Optional[np.ndarray]:
    path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
    if not os.path.exists(path) or is_lfs_pointer(path):
        return None
    seg = nib.load(path).get_fdata(dtype=np.float32).astype(np.int64)
    seg[seg == 4] = 3
    return seg


def zscore(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    for i in range(out.shape[0]):
        channel = out[i]
        std = channel.std()
        if std > 1e-6:
            out[i] = (channel - channel.mean()) / std
    return out


@torch.no_grad()
def predict_case(model, image: np.ndarray, window_size, stride, device) -> tuple:
    tensor = torch.from_numpy(image[None]).float().to(device)
    logits = sliding_window_logits(model, tensor, window_size=window_size, stride=stride)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)

    uncertainty = None
    if getattr(model, "evcal", None) is not None:
        probs = model.evcal.predictive_probs(logits)
        uncertainty = model.evcal(logits)[0, 0].cpu().numpy()
    return pred, uncertainty, logits


def pick_slice(mask: np.ndarray) -> int:
    counts = [mask[:, :, z].sum() for z in range(mask.shape[2])]
    return int(np.argmax(counts)) if max(counts) > 0 else mask.shape[2] // 2


def draw_overlay(image: np.ndarray, pred: np.ndarray, seg: Optional[np.ndarray],
                 slice_idx: int, path: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, channel, name in zip(
        axes[:4], range(image.shape[0]), ["FLAIR", "T1", "T1ce", "T2"]
    ):
        ax.imshow(np.rot90(image[channel, :, :, slice_idx]), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_prediction(image: np.ndarray, pred: np.ndarray, seg: Optional[np.ndarray],
                    slice_idx: int, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    flair = np.rot90(image[0, :, :, slice_idx])
    for ax, title in zip(axes, ["Ground truth", "KnowNet prediction"]):
        ax.imshow(flair, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    if seg is not None:
        axes[0].contour(np.rot90(seg[:, :, slice_idx]), levels=[0.5, 1.5, 2.5],
                        colors=["#4CAF50", "#FFC107", "#F44336"], linewidths=1.2)
    axes[1].contour(np.rot90(pred[:, :, slice_idx]), levels=[0.5, 1.5, 2.5],
                    colors=["#4CAF50", "#FFC107", "#F44336"], linewidths=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_uncertainty(image: np.ndarray, uncertainty: Optional[np.ndarray],
                     slice_idx: int, path: str) -> None:
    if uncertainty is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.rot90(image[0, :, :, slice_idx]), cmap="gray")
    axes[0].set_title("FLAIR")
    axes[0].axis("off")
    heat = axes[1].imshow(np.rot90(uncertainty[:, :, slice_idx]), cmap="magma")
    axes[1].set_title("EvCal uncertainty U")
    axes[1].axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="KnowNet batch inference")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=os.path.join("weights", "best_checkpoint.pth"))
    parser.add_argument("--input-dir", default="predict")
    parser.add_argument("--output-dir", default=os.path.join("predict", "results"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-sliding-window", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device", "auto"))
    print(f"[env] {device_summary(device)} | selected={device}")

    evaluation_config = config.get("evaluation", {})
    window_size = tuple(evaluation_config.get("window_size", (128, 128, 128)))
    stride = tuple(evaluation_config.get("stride", (64, 64, 64)))

    model = build_model(config)
    if os.path.exists(args.checkpoint):
        missing, unexpected = load_checkpoint(model, args.checkpoint, device)
        print(f"[checkpoint] {args.checkpoint} (missing={len(missing)})")
    else:
        print(
            f"[checkpoint] WARNING: {args.checkpoint} not found; using random weights. "
            "See weight/download.txt for the released checkpoint."
        )
    model.to(device).eval()

    case_dirs = sorted(
        os.path.join(args.input_dir, name)
        for name in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, name))
        and name.startswith("BraTS")
    )
    if not case_dirs:
        raise SystemExit(f"No BraTS*_* case directories found under {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    hd95_fn = HD95Score()
    samples: List[Dict] = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        image = load_volume(case_dir, case_id)
        if image is None:
            continue
        image = zscore(image)
        seg = load_seg(case_dir, case_id)

        if args.no_sliding_window:
            crop = tuple(config["data"].get("crop_size", (128, 128, 128)))
            starts = [(s - c) // 2 for s, c in zip(image.shape[1:], crop)]
            slices = tuple(slice(max(0, s), max(0, s) + c) for s, c in zip(starts, crop))
            volume = image[(slice(None),) + slices]
            with torch.no_grad():
                logits = model(torch.from_numpy(volume[None]).float().to(device))[0]
            pred_crop = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)
            # Write the centre-crop prediction back into a full-size canvas so the
            # exported NIfTI, the overlays and the metrics share one geometry.
            pred = np.zeros(image.shape[1:], dtype=np.int16)
            pred[slices] = pred_crop
            uncertainty = None
        else:
            pred, uncertainty, _ = predict_case(model, image, window_size, stride, device)

        nib.save(
            nib.Nifti1Image(pred.astype(np.int16), np.eye(4)),
            os.path.join(args.output_dir, f"{case_id}_pred.nii.gz"),
        )

        slice_idx = pick_slice(seg if seg is not None else pred)
        draw_overlay(image, pred, seg, slice_idx,
                     os.path.join(args.output_dir, f"{case_id}_modalities.png"))
        draw_prediction(image, pred, seg, slice_idx,
                        os.path.join(args.output_dir, f"{case_id}_overlay.png"))
        draw_uncertainty(image, uncertainty, slice_idx,
                         os.path.join(args.output_dir, f"{case_id}_uncertainty.png"))

        info: Dict[str, float] = {"case_id": case_id}
        if seg is not None:
            pred_t = torch.from_numpy(pred)[None].long()
            seg_t = torch.from_numpy(seg)[None].long()
            info["dice_WT"] = region_dice(pred_t, seg_t)["WT"]
            info["dice_TC"] = region_dice(pred_t, seg_t)["TC"]
            info["dice_ET"] = region_dice(pred_t, seg_t)["ET"]
            info["hd95_mean"] = float(np.mean(hd95_fn(pred_t, seg_t)))
        if uncertainty is not None:
            info["mean_uncertainty"] = float(uncertainty.mean())
        samples.append(info)
        print(f"[predict] {case_id}: {info}")

    with open(os.path.join(args.output_dir, "samples_info.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[done] {len(samples)} case(s) written to {args.output_dir}")


if __name__ == "__main__":
    main()
