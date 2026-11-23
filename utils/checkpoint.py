"""Checkpoint loading with module-name remapping.

Some checkpoints use the alternative module names ``pyragate`` and
``ekd_calib`` and store a learnable ``ekd_calib.log_tau``.  These state
dictionaries are mapped onto the current ``prgate`` / ``evcal`` naming, where
the calibration temperature is kept as a buffer ``evcal.tau``.
"""

from typing import Tuple

import torch
import torch.nn as nn

# Longest prefixes first.
_LEGACY_PREFIXES = (
    ("pyragate.", "prgate."),
    ("ekd_calib.", "evcal."),
)


def extract_state_dict(checkpoint) -> dict:
    """Return the model ``state_dict`` from either a raw or a wrapped checkpoint."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        if isinstance(state, dict):
            return state
    return checkpoint


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """Rename legacy keys to the current ones."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in _LEGACY_PREFIXES:
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        if new_key == "evcal.log_tau":
            # learnable log-temperature -> fitted tau buffer
            value = torch.exp(torch.as_tensor(value)).clamp(0.01, 10.0)
            new_key = "evcal.tau"
        remapped[new_key] = value
    return remapped


def load_checkpoint(model: nn.Module, path: str, device="cpu",
                    strict: bool = False) -> Tuple[list, list]:
    """Load a checkpoint into ``model`` and return ``(missing, unexpected)``.

    With ``strict=False`` (the default) checkpoints without the PrGate feature
    branch or the EvCal ``tau`` buffer still load: those entries are left at
    their initialised values.
    """
    checkpoint = torch.load(path, map_location=device)
    state = remap_legacy_state_dict(extract_state_dict(checkpoint))
    return model.load_state_dict(state, strict=strict)
