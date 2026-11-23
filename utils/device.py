"""Device selection helpers.

KnowNet is device agnostic: ``auto`` resolves to CUDA, then Apple MPS, then
CPU, so the same code runs on a GPU server or a CPU/MPS workstation.
"""

import warnings

import torch


def mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def resolve_device(spec="auto") -> torch.device:
    """Resolve a config ``device`` value to a concrete :class:`torch.device`.

    ``"auto"`` (or ``None``) picks CUDA > MPS > CPU.  An explicit device that
    is not available falls back to CPU with a warning instead of crashing.
    """
    if isinstance(spec, torch.device):
        return spec
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(f"CUDA requested ({spec}) but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not mps_available():
        warnings.warn(f"MPS requested ({spec}) but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"
