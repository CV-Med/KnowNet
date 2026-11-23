"""EvCal tests: softplus evidence, U in (0, 1], detached weight, post-hoc tau."""

import torch
import torch.nn.functional as F

from models.innovation_module import EvCal


def test_evidence_is_softplus_and_probs_normalised():
    torch.manual_seed(0)
    evcal = EvCal(num_classes=4)
    logits = torch.randn(2, 4, 4, 4, 4)
    alpha = evcal.alpha(logits)
    assert torch.allclose(alpha, F.softplus(logits) + 1.0)
    probs = evcal.predictive_probs(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5)


def test_uncertainty_range_and_weight():
    evcal = EvCal(num_classes=4)
    logits = torch.randn(1, 4, 8, 8, 8) * 3.0
    uncertainty = evcal(logits)
    assert uncertainty.shape == (1, 1, 8, 8, 8)
    assert torch.all(uncertainty > 0) and torch.all(uncertainty <= 1.0)
    expected = 1.0 / (1.0 + uncertainty.mean())
    assert torch.allclose(evcal.get_u_weight(logits), expected)


def test_tau_is_buffer_and_keeps_argmax():
    evcal = EvCal(num_classes=4)
    assert len(list(evcal.parameters())) == 0
    logits = torch.randn(1, 4, 6, 6, 6)
    pred_before = evcal.predictive_probs(logits).argmax(dim=1)
    evcal.set_tau(0.82)
    assert torch.allclose(evcal.tau, torch.tensor(0.82))
    pred_after = evcal.predictive_probs(logits).argmax(dim=1)
    assert torch.equal(pred_before, pred_after)
    # tau only rescales the evidence concentration, i.e. it changes U.
    assert not torch.allclose(evcal(logits), evcal(logits, tau=torch.tensor(1.0)))


def test_fit_temperature_brier_recovers_sharpness():
    """Fitting tau on over-confident logits returns a tau on the search grid."""
    torch.manual_seed(0)
    num_voxels = 4000
    target = torch.randint(1, 4, (num_voxels,))
    logits = torch.randn(num_voxels, 4) * 0.5
    logits[torch.arange(num_voxels), target] += 4.0  # deliberately over-confident
    logits = logits.view(1, 4, 10, 20, 20)
    target = target.view(1, 10, 20, 20)

    evcal = EvCal(num_classes=4)
    tau_star, info = evcal.fit_temperature(logits, target, num=100)
    assert 0.05 <= tau_star <= 5.0
    assert info["best_brier"] <= info["brier"][0] + 1e-6
    assert torch.allclose(evcal.tau, torch.tensor(tau_star))
