"""Calibration utilities for the evidential head.

The evidential temperature ``tau`` is fitted on the validation split by
minimising the Brier score over a 500-point logarithmic grid on ``[0.05, 5.00]``.
The module also provides ECE, Brier, NLL, ACE and the AUROC for error detection
on tumour voxels.

All functions operate on batched tensors so they can be used on a validation
or test loader without materialising a whole dataset at once:

``logits``  -- ``(N, K, D, H, W)`` raw segmentation logits
``target``  -- ``(N, D, H, W)`` integer labels in ``[0, K-1]``

By default only tumour voxels (``target > 0``) are scored.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def evidential_probs(logits: torch.Tensor, tau=1.0,
                     num_classes: Optional[int] = None) -> torch.Tensor:
    """Predictive probabilities ``pi = (softplus(z/tau) + 1) / A``."""
    num_classes = num_classes or logits.shape[1]
    tau = torch.as_tensor(tau, device=logits.device, dtype=logits.dtype).clamp(min=1e-3)
    alpha = F.softplus(logits / tau) + 1.0
    return alpha / alpha.sum(dim=1, keepdim=True)


def _default_mask(target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return target > 0
    return mask.to(torch.bool)


def brier_score(probs: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> float:
    """Brier score summed over classes and averaged over the selected voxels."""
    mask = _default_mask(target, mask)
    if mask.sum() == 0:
        return float("nan")
    num_classes = probs.shape[1]
    one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).to(probs.dtype)
    squared = ((probs - one_hot) ** 2).sum(dim=1)
    return squared[mask].mean().item()


def negative_log_likelihood(probs: torch.Tensor, target: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> float:
    """Mean negative log-likelihood of the target class over selected voxels."""
    mask = _default_mask(target, mask)
    if mask.sum() == 0:
        return float("nan")
    num_classes = probs.shape[1]
    safe = probs.clamp(min=1e-12)
    nll = -safe.gather(1, target.clamp(0, num_classes - 1).unsqueeze(1)).squeeze(1)
    return nll[mask].mean().item()


def _confidence_and_correct(probs: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    conf = probs.max(dim=1).values
    correct = (probs.argmax(dim=1) == target).to(probs.dtype)
    return conf[mask], correct[mask]


def expected_calibration_error(probs: torch.Tensor, target: torch.Tensor,
                               num_bins: int = 10, mask: Optional[torch.Tensor] = None,
                               conf_min: float = 0.5) -> float:
    """ECE with equal-width bins over ``[conf_min, 1]`` (10 bins by default)."""
    mask = _default_mask(target, mask)
    conf, correct = _confidence_and_correct(probs, target, mask)
    if conf.numel() == 0:
        return float("nan")
    edges = torch.linspace(conf_min, 1.0, num_bins + 1, device=probs.device, dtype=probs.dtype)
    ece = conf.new_zeros(())
    total = conf.numel()
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf >= lo) & (conf < hi if i < num_bins - 1 else conf <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        ece = ece + count * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return (ece / total).item()


def adaptive_calibration_error(probs: torch.Tensor, target: torch.Tensor,
                               num_bins: int = 10, mask: Optional[torch.Tensor] = None) -> float:
    """ACE with equal-mass bins (10 deciles by default)."""
    mask = _default_mask(target, mask)
    conf, correct = _confidence_and_correct(probs, target, mask)
    if conf.numel() == 0:
        return float("nan")
    order = torch.argsort(conf)
    conf, correct = conf[order], correct[order]
    chunks = torch.chunk(torch.arange(conf.numel(), device=conf.device), num_bins)
    ace = conf.new_zeros(())
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        gap = (correct[chunk].mean() - conf[chunk].mean()).abs()
        ace = ace + chunk.numel() * gap
    return (ace / conf.numel()).item()


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the Mann-Whitney U statistic (no scikit-learn dependency).

    ``scores``: higher means "more likely positive"; ``labels``: 0/1.
    """
    scores = scores.reshape(-1).float()
    labels = labels.reshape(-1).bool()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)

    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    rank_sum_pos = ranks[labels].sum()
    return ((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item()


def error_detection_auroc(uncertainty: torch.Tensor, probs: torch.Tensor,
                          target: torch.Tensor, mask: Optional[torch.Tensor] = None,
                          use_uncertainty: bool = True) -> float:
    """AUROC of an error score (uncertainty or ``1 - max prob``)."""
    mask = _default_mask(target, mask)
    if mask.sum() == 0:
        return float("nan")
    if use_uncertainty:
        # ``U`` is a per-voxel scalar map: accept both (B, D, H, W) and
        # (B, 1, D, H, W) layouts.
        if uncertainty.dim() == target.dim() + 1 and uncertainty.shape[1] == 1:
            uncertainty = uncertainty.squeeze(1)
        scores = uncertainty[mask]
    else:
        scores = 1.0 - probs.max(dim=1).values[mask]
    errors = (probs.argmax(dim=1)[mask] != target[mask]).float()
    return binary_auroc(scores, errors)


@torch.no_grad()
def fit_temperature_brier(logits: torch.Tensor, target: torch.Tensor,
                          num_classes: Optional[int] = None,
                          tau_min: float = 0.05, tau_max: float = 5.0,
                          num: int = 500, mask: Optional[torch.Tensor] = None
                          ) -> Tuple[float, Dict[str, list]]:
    """Grid-search ``tau`` minimising the Brier score.

    Returns ``(tau_star, info)`` where ``info`` contains the grids for
    plotting the temperature curve.
    """
    num_classes = num_classes or logits.shape[1]
    # Build the grid on CPU: ``torch.logspace`` is not implemented for MPS.
    taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), num)
    briers = []
    for tau in taus:
        probs = evidential_probs(logits, float(tau.item()), num_classes=num_classes)
        briers.append(float(brier_score(probs, target, mask)))
    briers_t = torch.tensor(briers, dtype=torch.float32)
    best = int(torch.argmin(briers_t).item())
    tau_star = float(taus[best].item())
    return tau_star, {
        "tau": [float(t) for t in taus.cpu()],
        "brier": [float(b) for b in briers_t.cpu()],
        "best_index": best,
        "best_brier": float(briers_t[best].item()),
    }
