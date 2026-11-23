"""End-to-end pipeline tests on a tiny synthetic BraTS-format dataset.

These tests exercise real NIfTI I/O, the case-level split, the trainer, the
whole-volume evaluator and ``predict/predict.py`` without requiring the actual
BraTS download or a GPU.
"""

import copy
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODALITIES = ["flair", "t1", "t1ce", "t2"]


def _write_case(root: str, case_id: str, shape=(40, 40, 40), seed: int = 0):
    rng = np.random.RandomState(seed)
    case_dir = os.path.join(root, case_id)
    os.makedirs(case_dir, exist_ok=True)
    affine = np.eye(4)
    for modality in MODALITIES:
        volume = rng.randn(*shape).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, affine),
                 os.path.join(case_dir, f"{case_id}_{modality}.nii.gz"))
    seg = np.zeros(shape, dtype=np.int16)
    seg[10:25, 10:25, 10:25] = 2          # edema
    seg[12:20, 12:20, 12:20] = 1          # necrotic core
    seg[14:18, 14:18, 14:18] = 4          # enhancing tumor -> remapped to 3
    nib.save(nib.Nifti1Image(seg, affine),
             os.path.join(case_dir, f"{case_id}_seg.nii.gz"))


@pytest.fixture()
def tiny_dataset(tmp_path):
    root = tmp_path / "BraTS2021"
    root.mkdir()
    for i in range(6):
        _write_case(str(root), f"BraTS2021_{i:05d}", seed=i)
    return str(root)


def _tiny_config(base_config, data_root: str) -> dict:
    config = copy.deepcopy(base_config)
    config["device"] = "cpu"
    config["seed"] = 0
    config["data"].update({
        "data_root": data_root,
        "crop_size": [32, 32, 32],
        "split_ratio": [0.66, 0.17, 0.17],
        "num_workers": 0,
        "tumor_centered_crop": True,
    })
    config["model"].update({
        "enc_features": [8, 16, 32, 64, 96],
        "bottleneck_channels": 96,
        "coord_hidden": 8,
        "gating_variant": "coord_feature",
    })
    config["training"].update({
        "epochs": 1, "batch_size": 1, "accum_steps": 1, "val_every": 1,
        "warmup_epochs": 1, "early_stop_patience": 2,
    })
    config["evaluation"].update({
        "window_size": [32, 32, 32],
        "stride": [16, 16, 16],
        "validation_sliding_window": True,
        "compute_hd95": True,
        "sliding_window": True,
    })
    config["calibration"].update({"max_voxels": 4000, "max_voxels_per_case": 1000,
                                  "tau_grid_points": 20})
    config["ablation"]["finetune_epochs"] = 1
    return config


def test_dataset_shapes_and_labels(tiny_dataset, base_config):
    from data.dataset import get_datasets

    config = _tiny_config(base_config, tiny_dataset)
    train, val, test = get_datasets(config)
    image, target = train[0]
    assert image.shape == (4, 32, 32, 32)
    assert target.shape == (32, 32, 32)
    assert image.dtype == torch.float32 and target.dtype == torch.int64
    assert set(target.unique().tolist()) <= {0, 1, 2, 3}  # label 4 remapped
    assert val[0][0].shape[1:] == (40, 40, 40)            # full volume for val
    assert test[0][0].shape[1:] == (40, 40, 40)


def test_train_epoch_and_evaluate(tiny_dataset, tmp_path, monkeypatch, base_config):
    """One training epoch + whole-volume evaluation + tau fitting."""
    from data.dataset import get_datasets
    from engines.evaluator import Evaluator
    from engines.trainer import Trainer
    from models import build_model

    config = _tiny_config(base_config, tiny_dataset)
    config["data"]["num_workers"] = 0
    monkeypatch.chdir(tmp_path)   # keep logs/ and weights/ inside tmp_path

    train, val, test = get_datasets(config)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config)
    trainer = Trainer(model, train_loader, val_loader, config,
                      checkpoint_dir=str(tmp_path / "weights"))
    trainer.fit(epochs=1, load_checkpoint=False)
    assert os.path.exists(tmp_path / "weights" / "latest_checkpoint.pth")

    evaluator = Evaluator(model, test_loader, config, device="cpu")
    results = evaluator.evaluate()
    for key in ("dice_WT", "dice_TC", "dice_ET", "hd95_mean", "num_cases"):
        assert key in results
    assert results["num_cases"] == len(test)
    assert results["num_cases"] >= 1
    assert 0.0 <= results["dice_mean"] <= 1.0

    # Post-hoc temperature fit.
    from utils.temperature import calibrate_model
    calibration = calibrate_model(
        config, model, val_loader, torch.device("cpu"),
        str(tmp_path / "weights" / "best_checkpoint_calibrated.pth"),
    )
    assert 0.05 <= calibration["tau"] <= 5.0
    assert os.path.exists(tmp_path / "weights" / "best_checkpoint_calibrated.pth")
    assert os.path.exists(tmp_path / "calibration.json")


def test_predict_script_end_to_end(tiny_dataset, tmp_path, base_config):
    config = _tiny_config(base_config, tiny_dataset)
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    output_dir = tmp_path / "predictions"

    command = [
        sys.executable, os.path.join(PROJECT_ROOT, "predict", "predict.py"),
        "--config", str(config_path),
        "--input-dir", tiny_dataset,
        "--output-dir", str(output_dir),
        "--checkpoint", str(tmp_path / "missing_checkpoint.pth"),
        "--device", "cpu",
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    predicted = sorted(os.listdir(output_dir))
    assert "samples_info.json" in predicted
    for case_id in ("BraTS2021_00000", "BraTS2021_00001"):
        assert f"{case_id}_pred.nii.gz" in predicted
        assert f"{case_id}_overlay.png" in predicted
    with open(output_dir / "samples_info.json") as f:
        samples = json.load(f)
    assert len(samples) == 6
    assert 0.0 <= samples[0]["dice_WT"] <= 1.0
