"""Calibration utilities: formulas, metrics and the post-hoc temperature fit."""

import numpy as np
import torch

from utils.calibration import (
    adaptive_calibration_error,
    binary_auroc,
    brier_score,
    error_detection_auroc,
    evidential_probs,
    expected_calibration_error,
    fit_temperature_brier,
    negative_log_likelihood,
)


def _one_hot_probs(labels: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes).permute(0, 4, 1, 2, 3).float()


def test_evidential_probs_are_softplus_based():
    logits = torch.randn(2, 4, 4, 4, 4)
    probs = evidential_probs(logits, tau=1.0)
    alpha = torch.nn.functional.softplus(logits) + 1.0
    assert torch.allclose(probs, alpha / alpha.sum(dim=1, keepdim=True), atol=1e-6)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)


def test_perfect_predictions_have_zero_ece_and_brier():
    labels = torch.randint(1, 4, (1, 6, 6, 6))
    probs = _one_hot_probs(labels)
    target = labels
    assert expected_calibration_error(probs, target) < 1e-6
    assert brier_score(probs, target) < 1e-6
    assert negative_log_likelihood(probs, target) < 1e-6


def test_confidence_gap_increases_ece():
    labels = torch.randint(1, 4, (1, 6, 6, 6))
    perfect = _one_hot_probs(labels)
    # A confident but wrong one-hot prediction is maximally miscalibrated.
    wrong = torch.roll(perfect, shifts=1, dims=1)
    assert expected_calibration_error(wrong, labels) > expected_calibration_error(perfect, labels)


def test_adaptive_calibration_error_between_zero_and_one():
    labels = torch.randint(1, 4, (1, 6, 6, 6))
    probs = torch.softmax(torch.randn(1, 4, 6, 6, 6), dim=1)
    value = adaptive_calibration_error(probs, labels)
    assert 0.0 <= value <= 1.0


def test_binary_auroc():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert binary_auroc(scores, torch.tensor([0, 0, 1, 1])) == 1.0
    assert binary_auroc(scores, torch.tensor([1, 1, 0, 0])) == 0.0
    assert binary_auroc(torch.ones(4), torch.tensor([0, 1, 0, 1])) == 0.5


def test_error_detection_auroc_uses_uncertainty():
    labels = torch.tensor([1, 1, 2, 2]).view(1, 1, 1, 4)
    uncertainty = torch.tensor([0.1, 0.1, 0.9, 0.9]).view(1, 1, 1, 1, 4)
    probs = torch.zeros(1, 4, 1, 1, 4)
    # First two voxels correct (class 1), last two wrong (predicted class 3).
    probs[0, 1, 0, 0] = torch.tensor([0.9, 0.9, 0.0, 0.0])
    probs[0, 3, 0, 0] = torch.tensor([0.0, 0.0, 0.9, 0.9])
    auc = error_detection_auroc(uncertainty, probs, labels)
    assert auc == 1.0


def test_fit_temperature_reduces_brier():
    torch.manual_seed(0)
    labels = torch.randint(1, 4, (1, 8, 8, 8))
    logits = torch.randn(1, 4, 8, 8, 8)
    logits[0, labels[0].unsqueeze(0), torch.arange(8).view(1, 8, 1),
           torch.arange(8).view(1, 1, 8)] += 6.0
    tau_star, info = fit_temperature_brier(logits, labels, num=100)
    brier_at_one = brier_score(evidential_probs(logits, 1.0), labels)
    assert 0.05 <= tau_star <= 5.0
    assert info["best_brier"] <= brier_at_one + 1e-6
    assert len(info["tau"]) == len(info["brier"]) == 100


def test_temperature_grid_is_logarithmic():
    labels = torch.randint(1, 4, (1, 4, 4, 4))
    logits = torch.randn(1, 4, 4, 4, 4)
    _, info = fit_temperature_brier(logits, labels, num=20)
    taus = np.array(info["tau"])
    assert taus[0] >= 0.05 - 1e-6 and taus[-1] <= 5.0 + 1e-6
    assert np.all(np.diff(taus) > 0)
