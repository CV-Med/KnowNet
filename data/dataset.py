"""BraTS dataset for KnowNet.

* case-level 8:1:1 training / validation / test split, shared by all runs;
* per-scan, per-channel z-score normalisation (no statistics shared across
  cases), label 4 remapped to 3;
* training uses 128^3 crops centred on a randomly chosen tumour voxel;
* validation and test keep the complete volume so that inference can use a
  sliding window and Dice / HD95 cover the whole tumour.
"""

import glob
import os
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import get_train_transforms, get_val_transforms


def _load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def _center_crop_at(data: np.ndarray, center: Sequence[int],
                    crop_size: Sequence[int]) -> np.ndarray:
    """Crop ``crop_size`` around ``center``, padding when the volume is smaller."""
    spatial = data.shape[-3:]
    crop = tuple(int(c) for c in crop_size)
    starts = []
    for size, target, c in zip(spatial, crop, center):
        start = int(round(c - target / 2.0))
        start = max(0, min(start, max(0, size - target)))
        starts.append(start)

    spatial_slices = [slice(s, s + t) for s, t in zip(starts, crop)]
    prepend = [slice(None)] if data.ndim > 3 else []
    out = data[tuple(prepend + spatial_slices)]
    current = out.shape[-3:]
    pads = [max(0, c - s) for c, s in zip(crop, current)]
    if any(pads):
        out = np.pad(out, [(0, 0)] * (data.ndim - 3) + [(p, 0) for p in pads],
                     mode="constant")
    return out


class BraTSDataset(Dataset):
    """BraTS NIfTI dataset (one case per item).

    Parameters
    ----------
    data_root : str
        Directory containing ``BraTS*_*/`` case folders.
    modalities : list of str
        Modality suffixes, e.g. ``["flair", "t1", "t1ce", "t2"]``.
    mode : {"train", "val", "test"}
        Selects the case-level split partition.
    crop_size : tuple
        Training crop size (also used by ``center_crop`` when
        ``full_volume=False`` for validation/test).
    split_ratio : (train, val, test)
        Fractions summing to 1.0 (default 0.8/0.1/0.1).
    transform : callable, optional
        ``transform(image=..., target=...)``-style pipeline.
    seed : int
        Seed of the case-level split (shared across runs).
    tumor_centered_crop : bool
        Sample the training crop around a random tumour voxel.
    full_volume : bool
        Keep the complete volume for validation/test (needed by sliding-window
        inference).  If ``False`` the volume is centre-cropped to ``crop_size``.
    """

    def __init__(self, data_root: str, modalities: List[str], mode: str = "train",
                 crop_size: Tuple[int, int, int] = (128, 128, 128),
                 split_ratio: Sequence[float] = (0.8, 0.1, 0.1),
                 transform=None, seed: int = 42,
                 tumor_centered_crop: bool = True,
                 full_volume: Optional[bool] = None,
                 train_ratio: Optional[float] = None,
                 case_glob: str = "BraTS*_*"):
        self.data_root = data_root
        self.modalities = list(modalities)
        self.mode = "val" if mode == "valid" else mode
        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"Unknown mode '{mode}', expected train/val/test")
        self.crop_size = tuple(int(c) for c in crop_size)
        self.transform = transform
        self.tumor_centered_crop = tumor_centered_crop
        self.full_volume = (self.mode != "train") if full_volume is None else bool(full_volume)
        self.split_ratio = split_ratio
        self.train_ratio = train_ratio
        self.case_glob = case_glob

        self.case_dirs = self._collect_cases()
        self.train_cases, self.val_cases, self.test_cases = self._split_cases(seed)
        if self.mode == "train":
            self.case_dirs = self.train_cases
        elif self.mode == "val":
            self.case_dirs = self.val_cases
        else:
            self.case_dirs = self.test_cases
        if not self.case_dirs:
            raise ValueError(
                f"Empty '{self.mode}' split ({len(self.train_cases)}/"
                f"{len(self.val_cases)}/{len(self.test_cases)} cases). "
                f"Check data.data_root and the split ratio."
            )

    # ------------------------------------------------------------------ data
    def _collect_cases(self) -> List[str]:
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(
                f"Dataset directory '{self.data_root}' does not exist. "
                f"See data/README.md for the expected layout and download links."
            )
        cases = sorted(
            d for d in glob.glob(os.path.join(self.data_root, self.case_glob))
            if os.path.isdir(d)
        )
        cases = [
            d for d in cases
            if os.path.exists(os.path.join(d, f"{os.path.basename(d)}_seg.nii.gz"))
        ]
        if not cases:
            raise FileNotFoundError(
                f"No BraTS cases found under '{self.data_root}' matching "
                f"'{self.case_glob}/*_seg.nii.gz'. See data/README.md."
            )
        return cases

    def _split_cases(self, seed: int):
        n = len(self.case_dirs)
        if self.split_ratio is None:
            ratio = (float(self.train_ratio or 0.8), 1.0 - float(self.train_ratio or 0.8), 0.0)
        else:
            ratio = tuple(float(r) for r in self.split_ratio)
        rng = np.random.RandomState(seed)
        order = rng.permutation(n)
        n_train = int(n * ratio[0])
        n_val = int(round(n * ratio[1]))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        idx_train = order[:n_train]
        idx_val = order[n_train:n_train + n_val]
        idx_test = order[n_train + n_val:]
        cases = self.case_dirs
        return ([cases[i] for i in idx_train],
                [cases[i] for i in idx_val],
                [cases[i] for i in idx_test])

    @property
    def case_ids(self) -> List[str]:
        return [os.path.basename(d) for d in self.case_dirs]

    # ------------------------------------------------------------- item load
    def __len__(self) -> int:
        return len(self.case_dirs)

    def _crop(self, image: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.full_volume:
            return image, target
        if self.mode == "train" and self.tumor_centered_crop:
            foreground = np.argwhere(target > 0)
            if len(foreground) > 0:
                center = foreground[np.random.randint(len(foreground))]
            else:
                center = np.array(target.shape) // 2
        else:
            center = np.array(target.shape) // 2
        return (
            _center_crop_at(image, center, self.crop_size),
            _center_crop_at(target, center, self.crop_size),
        )

    def __getitem__(self, idx: int):
        case_dir = self.case_dirs[idx]
        case_id = os.path.basename(case_dir)

        volumes = [
            _load_nifti(os.path.join(case_dir, f"{case_id}_{mod}.nii.gz"))
            for mod in self.modalities
        ]
        image = np.stack(volumes, axis=0)

        target = _load_nifti(os.path.join(case_dir, f"{case_id}_seg.nii.gz"))
        target = target.astype(np.int64)
        target[target == 4] = 3

        if self.transform is not None:
            augmented = self.transform(image=image, target=target)
            image, target = augmented["image"], augmented["target"]

        image, target = self._crop(image, target)

        image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        target = torch.from_numpy(np.ascontiguousarray(target, dtype=np.int64))
        return image, target


def get_datasets(config: dict):
    """Build the train / val / test datasets from a config dict."""
    data_config = config["data"]
    common = dict(
        data_root=data_config["data_root"],
        modalities=data_config["modalities"],
        crop_size=tuple(data_config.get("crop_size", (128, 128, 128))),
        split_ratio=tuple(data_config.get("split_ratio", (0.8, 0.1, 0.1))),
        seed=data_config.get("split_seed", config.get("seed", 42)),
        tumor_centered_crop=data_config.get("tumor_centered_crop", True),
        case_glob=data_config.get("case_glob", "BraTS*_*"),
    )
    train_dataset = BraTSDataset(
        mode="train", transform=get_train_transforms(), full_volume=False, **common
    )
    val_dataset = BraTSDataset(
        mode="val", transform=get_val_transforms(), full_volume=True, **common
    )
    test_dataset = BraTSDataset(
        mode="test", transform=get_val_transforms(), full_volume=True, **common
    )
    return train_dataset, val_dataset, test_dataset
