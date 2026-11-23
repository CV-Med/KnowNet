"""Sliding-window inference tests (whole-volume protocol)."""

import torch

from utils.inference import center_crop, sliding_window_logits


def test_single_window_equals_direct_forward(small_model):
    volume = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        direct = small_model(volume)[0]
        sliding = sliding_window_logits(
            small_model, volume, window_size=(32, 32, 32), stride=(16, 16, 16)
        )
    assert sliding.shape == direct.shape
    assert torch.allclose(sliding, direct, atol=1e-5)


def test_multi_window_preserves_shape(small_model):
    volume = torch.randn(1, 4, 40, 36, 44)
    with torch.no_grad():
        logits = sliding_window_logits(
            small_model, volume, window_size=(32, 32, 32), stride=(16, 16, 16)
        )
    assert logits.shape == (1, 4, 40, 36, 44)
    assert torch.isfinite(logits).all()


def test_small_volume_is_padded(small_model):
    # 24^3 < 32^3 window: the helper must pad instead of crashing.
    volume = torch.randn(1, 4, 24, 24, 24)
    with torch.no_grad():
        logits = sliding_window_logits(
            small_model, volume, window_size=(32, 32, 32), stride=(16, 16, 16)
        )
    assert logits.shape == (1, 4, 24, 24, 24)


def test_center_crop():
    volume = torch.arange(1 * 4 * 40 * 36 * 44, dtype=torch.float32).view(1, 4, 40, 36, 44)
    cropped = center_crop(volume, (32, 32, 32))
    assert cropped.shape == (1, 4, 32, 32, 32)
    # Central crop of a smaller volume keeps the available extent.
    small = volume[:, :, :20, :20, :20]
    assert center_crop(small, (32, 32, 32)).shape == (1, 4, 20, 20, 20)
