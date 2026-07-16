import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import glob
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import build_model


MODALITIES = ["flair", "t1", "t1ce", "t2"]
MODALITY_NAMES = ["Flair", "T1", "T1ce", "T2"]
LABEL_CMAP = ListedColormap(['black', 'red', 'green', 'blue'])
LABEL_NAMES = {0: "Background", 1: "NEC", 2: "ED", 3: "ET"}
CROP_SIZE = (128, 128, 128)


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


def per_channel_zscore(image):
    c = image.shape[0]
    out = image.copy()
    for i in range(c):
        ch = out[i]
        mean, std = ch.mean(), ch.std()
        if std > 1e-6:
            out[i] = (ch - mean) / std
    return out


def crop_center(data, crop_size):
    d, h, w = data.shape[-3:]
    cd, ch, cw = crop_size
    ds = max(0, (d - cd) // 2)
    hs = max(0, (h - ch) // 2)
    ws = max(0, (w - cw) // 2)
    return data[..., ds:ds + cd, hs:hs + ch, ws:ws + cw]


def find_tumor_center(target, axis=0):
    nonzero = np.sum(target > 0, axis=(1, 2)) if axis == 0 else \
              np.sum(target > 0, axis=(0, 2)) if axis == 1 else \
              np.sum(target > 0, axis=(0, 1))
    if nonzero.max() == 0:
        return target.shape[axis] // 2
    return int(np.argmax(nonzero))


def get_slice(data, axis, idx):
    if axis == 0:
        return data[idx, :, :]
    elif axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


def normalize_for_display(img_slice):
    img = img_slice.astype(np.float32)
    vmin, vmax = np.percentile(img, [1, 99])
    if vmax - vmin < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)


@torch.no_grad()
def predict_volume(model, image_np, device):
    image = per_channel_zscore(image_np)
    image = crop_center(image, CROP_SIZE)
    tensor = torch.from_numpy(image.copy()).unsqueeze(0).to(device)
    logits, aux_outputs, uncertainty = model(tensor)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred


def visualize_sample(sample_id, case_dir, pred, target, output_dir):
    volumes = []
    for mod in MODALITIES:
        mod_path = os.path.join(case_dir, f"{sample_id}_{mod}.nii.gz")
        volumes.append(load_nifti(mod_path))
    image = np.stack(volumes, axis=0)

    target[target == 4] = 3

    z_axis = 2
    center = find_tumor_center(target, axis=z_axis)
    total_slices = target.shape[z_axis]
    offset = max(5, int(total_slices * 0.2))
    slice_indices = [
        max(0, center - offset),
        center,
        min(total_slices - 1, center + offset),
    ]

    image_cropped = crop_center(image, CROP_SIZE)

    fig, axes_grid = plt.subplots(3, 6, figsize=(24, 13))
    fig.suptitle(f"{sample_id} — Segmentation Prediction", fontsize=18, fontweight='bold', y=0.98)

    for row, sl_idx in enumerate(slice_indices):
        for col, (data_source, title) in enumerate([
            (image_cropped[0], "Flair"),
            (image_cropped[1], "T1"),
            (image_cropped[2], "T1ce"),
            (image_cropped[3], "T2"),
            (target, "Ground Truth"),
            (pred, "Prediction"),
        ]):
            ax = axes_grid[row, col]
            sl = get_slice(data_source, z_axis, sl_idx)

            if col < 4:
                ax.imshow(normalize_for_display(sl), cmap='gray', aspect='auto')
            elif col == 4:
                ax.imshow(sl, cmap=LABEL_CMAP, vmin=0, vmax=3, aspect='auto')
            else:
                ax.imshow(sl, cmap=LABEL_CMAP, vmin=0, vmax=3, aspect='auto')

            if row == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"Slice {sl_idx}", fontsize=12)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, f"{sample_id}_prediction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    fig2, axes2 = plt.subplots(3, 3, figsize=(14, 13))
    fig2.suptitle(f"{sample_id} — Overlay (Prediction on Flair)", fontsize=16, fontweight='bold', y=0.98)

    for row, sl_idx in enumerate(slice_indices):
        flair_sl = normalize_for_display(get_slice(image_cropped[0], z_axis, sl_idx))
        gt_sl = get_slice(target, z_axis, sl_idx)
        pred_sl = get_slice(pred, z_axis, sl_idx)

        ax_flair = axes2[row, 0]
        ax_flair.imshow(flair_sl, cmap='gray', aspect='auto')
        ax_flair.set_title("Flair" if row == 0 else "", fontsize=13, fontweight='bold')
        ax_flair.set_ylabel(f"Slice {sl_idx}", fontsize=12)
        ax_flair.axis('off')

        ax_gt = axes2[row, 1]
        ax_gt.imshow(flair_sl, cmap='gray', aspect='auto')
        gt_mask = gt_sl > 0
        gt_overlay = np.zeros((*gt_sl.shape, 4))
        for label in [1, 2, 3]:
            mask = gt_sl == label
            if mask.any():
                if label == 1:
                    gt_overlay[mask] = [1, 0, 0, 0.45]
                elif label == 2:
                    gt_overlay[mask] = [0, 1, 0, 0.45]
                else:
                    gt_overlay[mask] = [0, 0, 1, 0.45]
        ax_gt.imshow(gt_overlay)
        ax_gt.set_title("GT Overlay" if row == 0 else "", fontsize=13, fontweight='bold')
        ax_gt.axis('off')

        ax_pred = axes2[row, 2]
        ax_pred.imshow(flair_sl, cmap='gray', aspect='auto')
        pred_overlay = np.zeros((*pred_sl.shape, 4))
        for label in [1, 2, 3]:
            mask = pred_sl == label
            if mask.any():
                if label == 1:
                    pred_overlay[mask] = [1, 0, 0, 0.45]
                elif label == 2:
                    pred_overlay[mask] = [0, 1, 0, 0.45]
                else:
                    pred_overlay[mask] = [0, 0, 1, 0.45]
        ax_pred.imshow(pred_overlay)
        ax_pred.set_title("Pred Overlay" if row == 0 else "", fontsize=13, fontweight='bold')
        ax_pred.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    overlay_path = os.path.join(output_dir, f"{sample_id}_overlay.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  Saved: {overlay_path}")


def compute_region_dice(pred, target):
    pred = pred.copy()
    target = target.copy()
    target[target == 4] = 3
    smooth = 1e-6
    results = {}
    regions = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}
    for name, labels in regions.items():
        p = np.isin(pred, labels).astype(np.float32)
        t = np.isin(target, labels).astype(np.float32)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        results[name] = (2.0 * inter + smooth) / (union + smooth)
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)

    config = load_config("configs/config.yaml")
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config)
    model = model.to(device)
    ckpt_path = "weights/best_checkpoint.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}, using random weights")
    model.eval()

    sample_dirs = sorted(glob.glob(os.path.join(script_dir, "BraTS2021_*")))
    if not sample_dirs:
        print("No BraTS2021 samples found in predict/")
        return

    print(f"Found {len(sample_dirs)} samples to predict")
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    info = {"samples": []}

    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        print(f"\nProcessing {sample_id}...")

        target_path = os.path.join(sample_dir, f"{sample_id}_seg.nii.gz")
        target = load_nifti(target_path).astype(np.int64)

        volumes = []
        for mod in MODALITIES:
            mod_path = os.path.join(sample_dir, f"{sample_id}_{mod}.nii.gz")
            volumes.append(load_nifti(mod_path))
        image = np.stack(volumes, axis=0)

        pred = predict_volume(model, image, device)
        pred_nii = pred.copy()
        pred_nii[pred_nii == 4] = 3

        original_shape = target.shape
        target_cropped = crop_center(target, CROP_SIZE)

        region_metrics = compute_region_dice(pred, target_cropped)

        visualize_sample(sample_id, sample_dir, pred, target_cropped, output_dir)

        info["samples"].append({
            "sample_id": sample_id,
            "original_shape": list(original_shape),
            "cropped_shape": list(CROP_SIZE),
            "region_dice": region_metrics,
            "prediction_file": f"results/{sample_id}_prediction.png",
            "overlay_file": f"results/{sample_id}_overlay.png",
        })
        print(f"  Dice — WT: {region_metrics['WT']:.4f}, TC: {region_metrics['TC']:.4f}, ET: {region_metrics['ET']:.4f}")

    info_path = os.path.join(script_dir, "samples_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nAll done! Results saved to {output_dir}/")
    print(f"Sample info: {info_path}")


if __name__ == "__main__":
    main()
