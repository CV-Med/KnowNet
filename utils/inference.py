"""Whole-volume inference helpers.

Segmentation is evaluated on the complete MRI volume with a sliding window
(default 128^3, stride 64) instead of a single centre crop.  Keeping the
aggregation here (rather than in the evaluator) lets training, testing and the
prediction script share the same inference path.
"""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_tuple(values: Sequence[int]) -> Tuple[int, int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values * 3
    if len(values) != 3:
        raise ValueError(f"Expected 1 or 3 spatial values, got {values}")
    return values


def _pad_volume(volume: torch.Tensor, window: Tuple[int, int, int],
                stride: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Pad the end of each spatial axis so sliding windows tile the volume."""
    spatial = volume.shape[2:]
    pad = []
    for size, win, step in zip(reversed(spatial), reversed(window), reversed(stride)):
        if size < win:
            extra = win - size
        else:
            remainder = (size - win) % step
            extra = (step - remainder) % step
        pad.extend([0, extra])
    if any(pad):
        volume = F.pad(volume, pad, mode="constant", value=0.0)
    return volume, tuple(volume.shape[2:])


def _window_starts(size: int, window: int, stride: int) -> list:
    if size == window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


@torch.no_grad()
def sliding_window_logits(model, volume: torch.Tensor,
                          window_size=(128, 128, 128),
                          stride=(64, 64, 64)) -> torch.Tensor:
    """Run ``model`` over a full volume and average the window logits.

    Parameters
    ----------
    model : nn.Module
        Segmentation model returning ``(logits, aux_outputs, uncertainty)``.
    volume : torch.Tensor
        Normalised input of shape ``(1, C, D, H, W)``.  It is padded internally
        when a spatial size is smaller than the window.
    window_size, stride : sequence of 3 ints
        Sliding-window geometry (default 128^3 / 64).

    Returns
    -------
    torch.Tensor
        Logits of shape ``(1, K, D, H, W)`` (original, unpadded size).
    """
    if volume.dim() != 5 or volume.shape[0] != 1:
        raise ValueError(f"Expected volume of shape (1, C, D, H, W), got {tuple(volume.shape)}")

    window = _as_tuple(window_size)
    step = _as_tuple(stride)
    original = tuple(volume.shape[2:])
    padded, padded_size = _pad_volume(volume, window, step)

    num_classes = None
    accum = None
    counts = None
    for d in _window_starts(padded_size[0], window[0], step[0]):
        for h in _window_starts(padded_size[1], window[1], step[1]):
            for w in _window_starts(padded_size[2], window[2], step[2]):
                patch = padded[:, :, d:d + window[0], h:h + window[1], w:w + window[2]]
                logits = model(patch)[0]
                if accum is None:
                    num_classes = logits.shape[1]
                    accum = torch.zeros(
                        (1, num_classes) + padded_size, device=volume.device, dtype=torch.float32
                    )
                    counts = torch.zeros(
                        (1, 1) + padded_size, device=volume.device, dtype=torch.float32
                    )
                accum[:, :, d:d + window[0], h:h + window[1], w:w + window[2]] += logits.float()
                counts[:, :, d:d + window[0], h:h + window[1], w:w + window[2]] += 1

    accum = accum / counts.clamp(min=1.0)
    return accum[:, :, :original[0], :original[1], :original[2]]


def center_crop(volume: torch.Tensor, crop_size=(128, 128, 128)) -> torch.Tensor:
    """Central crop used when full-volume evaluation is disabled."""
    crop = _as_tuple(crop_size)
    spatial = volume.shape[2:]
    slices = [slice(None), slice(None)]
    for size, target in zip(spatial, crop):
        start = max(0, (size - target) // 2)
        slices.append(slice(start, start + target))
    return volume[tuple(slices)]
