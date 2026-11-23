"""Knowledge-driven modules for KnowNet.

* :class:`BoundGATE` -- multi-scale boundary gradient gating at the stage-2 skip.
* :class:`PrGate` -- feature-conditioned multiscale gating.  The gate combines a
  patient-invariant coordinate logit (a shared anatomical prior) with an
  image-conditioned feature logit, so the three dilation branches are fused with
  per-voxel, per-patient weights.
* :class:`EvCal` -- temperature-scaling evidential calibration.  During training
  the evidence is computed with a fixed ``tau = 1`` and the uncertainty is turned
  into a detached auxiliary weight.  After training a single scalar temperature
  is fitted on the validation split by minimising the Brier score; the fitted
  value is applied only at inference and leaves the argmax unchanged.

Legacy names ``PyraGATE`` and ``EKDCalib`` are kept as aliases for checkpoint and
script compatibility.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax_over_scales(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the scale dimension (dim=1).

    Implemented explicitly instead of ``torch.softmax`` because the 5-D
    ``softmax`` kernel is not available for every memory format on Apple MPS
    (``RuntimeError: Invalid memory formatChannelsLast3d``).
    """
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp = logits.exp()
    return exp / exp.sum(dim=1, keepdim=True)


class Gradient3D(nn.Module):
    """Gaussian-smoothed 3-D gradient magnitude, reduced over channels."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        center = kernel_size // 2
        x = torch.arange(kernel_size).float() - center
        y = torch.arange(kernel_size).float() - center
        z = torch.arange(kernel_size).float() - center
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        g = torch.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        gx = (-xx * g).unsqueeze(0).unsqueeze(0)
        gy = (-yy * g).unsqueeze(0).unsqueeze(0)
        gz = (-zz * g).unsqueeze(0).unsqueeze(0)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)
        self.register_buffer("gz", gz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_flat = x.reshape(b * c, 1, d, h, w)
        pad = self.gx.shape[2] // 2
        grad_x = F.conv3d(x_flat, self.gx, padding=pad)
        grad_y = F.conv3d(x_flat, self.gy, padding=pad)
        grad_z = F.conv3d(x_flat, self.gz, padding=pad)
        grad = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2).sqrt()
        grad = grad.view(b, c, d, h, w)
        return grad.max(dim=1, keepdim=True).values


class BoundGATE(nn.Module):
    """BGate: multi-scale boundary gradient -> Sigmoid gating at stage-2 skip.

    ``s2_gated = s2 * sigma(Conv1x1([s2, g1_up, g2]))``.

    Set ``capture_gate = True`` to keep the last gate map detached for
    visualisation.
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.grad1 = Gradient3D(sigma=1.0)
        self.grad2 = Gradient3D(sigma=2.0)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels + 2, in_channels, 1),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.Sigmoid(),
        )
        self.capture_gate = False
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        g1 = self.grad1(s1)
        g2 = self.grad2(s2)
        g1_up = F.interpolate(g1, size=s2.shape[2:], mode="trilinear", align_corners=False)
        gate = self.gate_conv(torch.cat([s2, g1_up, g2], dim=1))
        if self.capture_gate:
            self.last_gate = gate.detach()
        return s2 * gate


class PrGate(nn.Module):
    """PrGate: feature-conditioned multiscale gating.

    Three depthwise tri-dilation branches produce ``F_i`` (i = 1, 2, 3) with
    dilation rates ``d in [1, 2, 3]``.  The fusion weights are

    ``beta(c_v, F_v) = softmax_i( W2 h(c_v) + b2 + Conv1x1(F)_v )``

    where ``h`` is the coordinate MLP (shared anatomical prior) and the 1x1x1
    convolution is the image-conditioned evidence branch.  The module returns
    the residual fusion ``F + sum_i beta_i * F_i``.

    ``gating_variant`` selects one of five gating designs:

    ==================  ============================================
    ``uniform``         equal-weight fusion (no gate parameters)
    ``global``          one learned global weight per scale (3 params)
    ``coord``           shared coordinate prior only (115 params)
    ``feature``         image-conditioned evidence only (963 params)
    ``coord_feature``   full PrGate (1,078 params)
    ==================  ============================================
    """

    _VARIANTS = ("uniform", "global", "coord", "feature", "coord_feature")

    def __init__(self, channels: int = 256, dilation_rates: Optional[list] = None,
                 coord_hidden: int = 16, gating_variant: str = "coord_feature"):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 3]
        if gating_variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown gating_variant '{gating_variant}', expected one of {self._VARIANTS}"
            )
        self.channels = channels
        self.dilation_rates = dilation_rates
        self.num_scales = len(dilation_rates)
        self.gating_variant = gating_variant

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=d, dilation=d, groups=channels),
                nn.Conv3d(channels, channels, 1),
                nn.LeakyReLU(0.01, inplace=True),
            )
            for d in dilation_rates
        ])

        use_coord = gating_variant in ("coord", "coord_feature")
        use_feature = gating_variant in ("feature", "coord_feature")

        if use_coord:
            self.coord_mlp = nn.Sequential(
                nn.Linear(3, coord_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(coord_hidden, self.num_scales),
            )
        else:
            self.coord_mlp = None

        if use_feature:
            self.feature_logit = nn.Conv3d(channels, self.num_scales, 1)
        else:
            self.feature_logit = None

        if gating_variant == "global":
            self.global_logits = nn.Parameter(torch.zeros(self.num_scales))
        else:
            self.global_logits = None

        self.capture_alpha = False
        self.last_alpha: Optional[torch.Tensor] = None

    def _coord_logits(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        gd = torch.linspace(0.0, 1.0, d, device=x.device, dtype=x.dtype)
        gh = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
        gw = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_d, grid_h, grid_w = torch.meshgrid(gd, gh, gw, indexing="ij")
        coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)
        logits = self.coord_mlp(coords)             # (d, h, w, S)
        return logits.permute(3, 0, 1, 2).unsqueeze(0)  # (1, S, d, h, w)

    def _compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        variant = self.gating_variant
        if variant == "uniform":
            alpha = x.new_full((1, self.num_scales, 1, 1, 1), 1.0 / self.num_scales)
            return alpha.expand(x.shape[0], -1, x.shape[2], x.shape[3], x.shape[4])
        if variant == "global":
            weights = torch.softmax(self.global_logits, dim=0)
            return weights.view(1, self.num_scales, 1, 1, 1)
        if variant == "coord":
            logits = self._coord_logits(x)
        elif variant == "feature":
            logits = self.feature_logit(x)
        else:  # coord_feature
            logits = self._coord_logits(x) + self.feature_logit(x)
        return _softmax_over_scales(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._compute_alpha(x)
        out = 0
        for i, conv in enumerate(self.convs):
            out = out + alpha[:, i:i + 1] * conv(x)
        if self.capture_alpha:
            self.last_alpha = alpha.detach()
        return x + out


class EvCal(nn.Module):
    """EvCal: fitted temperature-scaling evidential calibration.

    Training (``tau = 1``)::

        e_c   = softplus(z_c / tau)             # evidence
        alpha = e + 1;  A = sum_c alpha_c       # concentration
        pi    = alpha / A                       # predictive probabilities
        U     = K / A                           # uncertainty, in (0, 1]

    The uncertainty is used by :class:`losses.loss.DeepSupervisionLoss` as a
    detached auxiliary weight ``w_U = 1 / (1 + mean(U))``, so it does not enter
    the segmentation gradient path.

    Inference: a single scalar ``tau`` (a buffer, not a parameter) is fitted
    after training on the validation split by minimising the Brier score over
    a 500-point logarithmic grid on ``[0.05, 5.00]``.  ``tau`` only rescales the
    evidence concentration and leaves the argmax unchanged.
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("tau", torch.tensor(1.0))

    def set_tau(self, tau: float) -> None:
        self.tau.fill_(float(tau))

    def _resolve_tau(self, tau: Optional[torch.Tensor]) -> torch.Tensor:
        if tau is None:
            tau = self.tau
        tau = torch.as_tensor(tau, device=self.tau.device, dtype=self.tau.dtype)
        return tau.clamp(min=1e-3)

    def evidence(self, logits: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        tau = self._resolve_tau(tau)
        return F.softplus(logits / tau)

    def alpha(self, logits: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.evidence(logits, tau) + 1.0

    def predictive_probs(self, logits: torch.Tensor,
                         tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        alpha = self.alpha(logits, tau)
        return alpha / alpha.sum(dim=1, keepdim=True)

    def forward(self, logits: torch.Tensor,
                tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        alpha = self.alpha(logits, tau)
        total = alpha.sum(dim=1, keepdim=True)
        return self.num_classes / total

    def get_u_weight(self, logits: torch.Tensor,
                     tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        uncertainty = self.forward(logits, tau)
        return 1.0 / (1.0 + uncertainty.mean())

    def fit_temperature(self, logits: torch.Tensor, target: torch.Tensor,
                        tau_min: float = 0.05, tau_max: float = 5.0,
                        num: int = 500, mask: Optional[torch.Tensor] = None):
        """Fit ``tau`` on the given (validation) logits by minimising Brier.

        Delegates to :func:`utils.calibration.fit_temperature_brier` and
        stores the fitted value in the ``tau`` buffer.
        """
        from utils.calibration import fit_temperature_brier

        tau_star, info = fit_temperature_brier(
            logits, target, num_classes=self.num_classes,
            tau_min=tau_min, tau_max=tau_max, num=num, mask=mask,
        )
        self.set_tau(tau_star)
        return tau_star, info


# ---------------------------------------------------------------------------
# Backward-compatible aliases for alternative module names.
# ---------------------------------------------------------------------------
PyraGATE = PrGate
EKDCalib = EvCal
EKD_CALIB = EvCal
BGate = BoundGATE
