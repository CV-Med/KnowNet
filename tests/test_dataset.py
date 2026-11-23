"""Dataset split / crop tests (case-level 8:1:1 protocol)."""

import os

import numpy as np
import pytest

from data.dataset import BraTSDataset, _center_crop_at


def _make_case_dirs(root, n_cases=10):
    for i in range(n_cases):
        case_dir = os.path.join(root, f"BraTS2021_{i:05d}")
        os.makedirs(case_dir, exist_ok=True)
        # Only the segmentation file is required for case discovery.
        open(os.path.join(case_dir, f"BraTS2021_{i:05d}_seg.nii.gz"), "w").close()


def test_split_is_8_1_1(tmp_path):
    root = str(tmp_path)
    _make_case_dirs(root, n_cases=10)
    common = dict(
        data_root=root,
        modalities=["flair", "t1", "t1ce", "t2"],
        crop_size=(16, 16, 16),
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    )
    train = BraTSDataset(mode="train", **common)
    val = BraTSDataset(mode="val", **common)
    test = BraTSDataset(mode="test", **common)

    assert (len(train), len(val), len(test)) == (8, 1, 1)
    train_ids, val_ids, test_ids = set(train.case_ids), set(val.case_ids), set(test.case_ids)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 10


def test_split_is_reproducible(tmp_path):
    _make_case_dirs(str(tmp_path), n_cases=10)
    kwargs = dict(
        data_root=str(tmp_path), modalities=["flair", "t1", "t1ce", "t2"],
        crop_size=(16, 16, 16), split_ratio=(0.8, 0.1, 0.1), seed=7,
    )
    first = BraTSDataset(mode="train", **kwargs).case_ids
    second = BraTSDataset(mode="train", **kwargs).case_ids
    assert first == second


def test_real_brats2021_split_sizes(tmp_path):
    """1251 cases must yield a 1000/125/126 split."""
    _make_case_dirs(str(tmp_path), n_cases=1251)
    kwargs = dict(
        data_root=str(tmp_path), modalities=["flair", "t1", "t1ce", "t2"],
        crop_size=(16, 16, 16), split_ratio=(0.8, 0.1, 0.1), seed=42,
    )
    counts = [len(BraTSDataset(mode=mode, **kwargs)) for mode in ("train", "val", "test")]
    assert counts == [1000, 125, 126], counts


def test_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        BraTSDataset(data_root="/nonexistent/BraTS2021", modalities=["flair"])


def test_center_crop_at():
    data = np.arange(4 * 10 * 20 * 30, dtype=np.float32).reshape(4, 10, 20, 30)
    cropped = _center_crop_at(data, center=(5, 10, 15), crop_size=(8, 8, 8))
    assert cropped.shape == (4, 8, 8, 8)

    small = np.ones((4, 6, 6, 6), dtype=np.float32)
    padded = _center_crop_at(small, center=(3, 3, 3), crop_size=(8, 8, 8))
    assert padded.shape == (4, 8, 8, 8)
    assert padded[0, 0, 0, 0] == 0.0
