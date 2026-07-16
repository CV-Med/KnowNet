import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from data.transforms import get_train_transforms, get_val_transforms


class BraTSDataset(Dataset):
    def __init__(self, data_root: str, modalities: list, mode: str = "train",
                 crop_size=(128, 128, 128), train_ratio: float = 0.8, transform=None,
                 seed: int = 42):
        self.data_root = data_root
        self.modalities = modalities
        self.mode = mode
        self.crop_size = crop_size
        self.train_ratio = train_ratio
        self.transform = transform

        case_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS2021_*")))
        rng = np.random.RandomState(seed)
        rng.shuffle(case_dirs)
        split_idx = int(len(case_dirs) * train_ratio)
        if mode == "train":
            self.case_dirs = case_dirs[:split_idx]
        else:
            self.case_dirs = case_dirs[split_idx:]

    def __len__(self):
        return len(self.case_dirs)

    def _load_nifti(self, path: str) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return data

    def _crop_center(self, data: np.ndarray, crop_size) -> np.ndarray:
        d, h, w = data.shape[-3:]
        cd, ch, cw = crop_size
        ds = max(0, (d - cd) // 2)
        hs = max(0, (h - ch) // 2)
        ws = max(0, (w - cw) // 2)
        return data[..., ds:ds + cd, hs:hs + ch, ws:ws + cw]

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_id = os.path.basename(case_dir)

        volumes = []
        for mod in self.modalities:
            mod_path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
            vol = self._load_nifti(mod_path)
            volumes.append(vol)
        image = np.stack(volumes, axis=0)

        seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        target = self._load_nifti(seg_path).astype(np.int64)

        target[target == 4] = 3

        if self.transform is not None:
            augmented = self.transform(image=image, target=target)
            image = augmented["image"]
            target = augmented["target"]

        image = self._crop_center(image, self.crop_size)
        target = self._crop_center(target, self.crop_size)

        image = torch.from_numpy(image.copy())
        target = torch.from_numpy(target.copy()).long().squeeze(0)

        return image, target
