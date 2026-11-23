"""Architecture tests: parameter counts and gating designs."""

import pytest
import torch

from models import BrainTumorSegModel, build_model


def test_full_model_parameter_count(base_config):
    """The full model has about 30.0M parameters."""
    model = build_model(base_config)
    params = sum(p.numel() for p in model.parameters())
    assert 29.9e6 < params < 30.2e6, params


@pytest.mark.parametrize(
    "variant,expected_gate_params",
    [
        ("uniform", 0),
        ("global", 3),
        ("coord", 115),
        ("feature", 963),
        ("coord_feature", 1078),
    ],
)
def test_prgate_variant_parameter_counts(base_config, variant, expected_gate_params):
    """Gate parameters: coord 0.115K, feature 0.963K, both 1.078K."""
    config = {**base_config, "model": {**base_config["model"], "gating_variant": variant}}
    model = build_model(config)
    gate_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "prgate.coord_mlp" in name
        or "prgate.feature_logit" in name
        or "prgate.global_logits" in name
    )
    assert gate_params == expected_gate_params, (variant, gate_params)


@pytest.mark.parametrize("variant", ["uniform", "global", "coord", "feature", "coord_feature"])
def test_gating_variants_forward(small_config, tiny_volume, variant):
    config = {**small_config, "model": {**small_config["model"], "gating_variant": variant}}
    model = build_model(config).eval()
    with torch.no_grad():
        logits, aux, uncertainty = model(tiny_volume)
    assert logits.shape == (1, 4, 32, 32, 32)
    assert len(aux) == 4
    assert uncertainty is not None and uncertainty.shape == (1, 1, 32, 32, 32)


def test_prgate_alpha_is_patient_conditioned(small_config):
    """The feature logit branch must make alpha depend on the input features."""
    from models import build_model

    model = build_model(small_config).eval()
    model.set_capture(True)
    with torch.no_grad():
        model(torch.randn(1, 4, 32, 32, 32))
        alpha_a = model.prgate.last_alpha.clone()
        model(torch.randn(1, 4, 32, 32, 32) * 5.0)
        alpha_b = model.prgate.last_alpha.clone()
    assert alpha_a.shape[1] == 3
    assert not torch.allclose(alpha_a, alpha_b, atol=1e-4)


def test_disabled_modules(small_config, tiny_volume):
    model = build_model(small_config, disabled_modules=["BGate", "PrGate", "EvCal"]).eval()
    with torch.no_grad():
        logits, _, uncertainty = model(tiny_volume)
    assert logits.shape == (1, 4, 32, 32, 32)
    assert uncertainty is None

    # Legacy module names are accepted as aliases.
    legacy = build_model(
        small_config, disabled_modules=["BoundGATE", "PyraGATE", "EKD_CALIB"]
    ).eval()
    with torch.no_grad():
        _, _, uncertainty = legacy(tiny_volume)
    assert uncertainty is None


def test_backward_pass(small_config):
    model = build_model(small_config)
    model.train()
    volume = torch.randn(1, 4, 32, 32, 32)
    target = torch.randint(0, 4, (1, 32, 32, 32))
    logits, aux, uncertainty = model(volume)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert any(p.grad is not None for p in model.prgate.parameters())
    # EvCal has no trainable parameters: tau is a buffer fitted post hoc.
    assert len(list(model.evcal.parameters())) == 0
