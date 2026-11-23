"""Shared pytest fixtures: tiny model/config so the suite runs on CPU in ~30 s."""

import copy
import os
import sys

import pytest
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(scope="session")
def base_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def small_config(base_config) -> dict:
    """Same pipeline as the full model but narrow and cheap enough for CI."""
    config = copy.deepcopy(base_config)
    config["device"] = "cpu"
    config["data"]["crop_size"] = [32, 32, 32]
    config["model"]["enc_features"] = [8, 16, 32, 64, 96]
    config["model"]["bottleneck_channels"] = 96
    config["model"]["coord_hidden"] = 8
    config["evaluation"]["window_size"] = [32, 32, 32]
    config["evaluation"]["stride"] = [16, 16, 16]
    config["evaluation"]["validation_sliding_window"] = False
    return config


@pytest.fixture(scope="session")
def small_model(small_config):
    from models import build_model

    model = build_model(small_config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tiny_volume() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 4, 32, 32, 32)
