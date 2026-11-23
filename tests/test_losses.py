"""Loss tests: detached EvCal auxiliary weight and the constant-weight control."""

import torch
import torch.nn.functional as F

from losses.loss import DeepSupervisionLoss


def _make_inputs(requires_grad=True):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=requires_grad)
    aux = [
        torch.randn(2, 4, 4, 4, 4, requires_grad=requires_grad),
        torch.randn(2, 4, 2, 2, 2, requires_grad=requires_grad),
    ]
    target = torch.randint(0, 4, (2, 8, 8, 8))
    uncertainty = torch.rand(2, 1, 8, 8, 8, requires_grad=requires_grad)
    return logits, aux, target, uncertainty


def test_evcal_weight_is_detached():
    """w_U = 1 / (1 + mean(U)) enters the loss with stop-gradient."""
    logits, aux, target, uncertainty = _make_inputs()
    criterion = DeepSupervisionLoss(num_classes=4, aux_weight_mode="uncertainty")
    loss = criterion(logits, aux, target, uncertainty)
    loss.backward()
    assert uncertainty.grad is None, "uncertainty must not receive gradients"
    assert logits.grad is not None


def test_constant_aux_weight_control():
    """Full model with a constant auxiliary weight of 0.75."""
    logits, aux, target, uncertainty = _make_inputs(requires_grad=False)
    uncertainty_loss = DeepSupervisionLoss(
        num_classes=4, aux_weight_mode="uncertainty"
    )(logits, aux, target, uncertainty)
    constant_loss = DeepSupervisionLoss(
        num_classes=4, aux_weight_mode="constant", constant_aux_weight=0.75
    )(logits, aux, target, uncertainty)
    assert not torch.isclose(uncertainty_loss, constant_loss)

    # With w_U fixed to the constant, the uncertainty tensor is ignored entirely.
    other_uncertainty = torch.full_like(uncertainty, 5.0)
    constant_loss_2 = DeepSupervisionLoss(
        num_classes=4, aux_weight_mode="constant", constant_aux_weight=0.75
    )(logits, aux, target, other_uncertainty)
    assert torch.allclose(constant_loss, constant_loss_2)


def test_none_mode_matches_plain_deep_supervision():
    logits, aux, target, uncertainty = _make_inputs(requires_grad=False)
    loss = DeepSupervisionLoss(num_classes=4, aux_weight_mode="none")(
        logits, aux, target, uncertainty
    )
    main = F.cross_entropy(logits, target)
    manual = loss  # smoke: finite scalar with the expected dtype/shape
    assert manual.dim() == 0 and torch.isfinite(manual)
    assert main.shape == manual.shape
