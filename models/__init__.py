"""KnowNet model assembly: SegResNet backbone + BGate + PrGate + EvCal."""

from typing import List, Optional

import torch
import torch.nn as nn

from models.backbone import Encoder, Decoder
from models.innovation_module import BoundGATE, EvCal, PrGate

# Canonical module keys.  Legacy names (BoundGATE / PyraGATE / EKD_CALIB) and
# the short names (BGate / PrGate / EvCal) are both accepted.
MODULE_ALIASES = {
    "boundgate": "boundgate", "bgate": "boundgate",
    "pyragate": "prgate", "prgate": "prgate",
    "ekdcalib": "evcal", "ekd_calib": "evcal", "evcal": "evcal",
}


def _canonical_module_keys(names: Optional[List[str]]) -> set:
    keys = set()
    for name in names or []:
        key = MODULE_ALIASES.get(str(name).lower().replace("-", "_"))
        if key is None:
            raise ValueError(
                f"Unknown module '{name}'. Known modules: BoundGATE/BGate, "
                f"PyraGATE/PrGate, EKD_CALIB/EvCal."
            )
        keys.add(key)
    return keys


class BrainTumorSegModel(nn.Module):
    """SegResNet backbone with the three pluggable knowledge modules.

    ``disabled_modules`` accepts any mix of the legacy and short names,
    e.g. ``["BGate"]``, ``["PrGate", "EvCal"]`` or
    ``["BoundGATE", "PyraGATE", "EKD_CALIB"]``.  A disabled module is replaced
    by ``None`` (identity path).

    ``gating_variant`` selects one of five PrGate designs: ``uniform``,
    ``global``, ``coord``, ``feature`` or ``coord_feature`` (default, full
    model).
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 enc_features: list = None, bottleneck_channels: int = 320,
                 dilation_rates: list = None, coord_hidden: int = 16,
                 disabled_modules: Optional[List[str]] = None,
                 gating_variant: str = "coord_feature"):
        super().__init__()
        if enc_features is None:
            enc_features = [32, 64, 128, 256, 320]
        if dilation_rates is None:
            dilation_rates = [1, 2, 3]

        disabled = _canonical_module_keys(disabled_modules)
        self.disabled_modules = sorted(disabled)
        self.gating_variant = gating_variant

        self.encoder = Encoder(in_channels, enc_features)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(enc_features[-1], bottleneck_channels, 3, padding=1),
            nn.GroupNorm(min(8, bottleneck_channels), bottleneck_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder = Decoder(enc_features, bottleneck_channels, num_classes)

        self.boundgate = (
            None if "boundgate" in disabled
            else BoundGATE(in_channels=enc_features[1])
        )
        self.prgate = (
            None if "prgate" in disabled
            else PrGate(bottleneck_channels, dilation_rates, coord_hidden,
                        gating_variant=gating_variant)
        )
        self.evcal = (
            None if "evcal" in disabled
            else EvCal(num_classes=num_classes)
        )

    # Read-only aliases kept for backward compatibility.
    @property
    def pyragate(self):
        return self.prgate

    @property
    def ekd_calib(self):
        return self.evcal

    def set_capture(self, flag: bool = True) -> None:
        """Enable/disable storing detached gate/scale maps for visualisation."""
        if self.boundgate is not None:
            self.boundgate.capture_gate = flag
        if self.prgate is not None:
            self.prgate.capture_alpha = flag

    def forward(self, x: torch.Tensor):
        s1, s2, s3, s4, s5, p5 = self.encoder(x)

        if self.boundgate is not None:
            s2_gated = self.boundgate(s1, s2)
        else:
            s2_gated = s2

        bottleneck = self.bottleneck(p5)

        if self.prgate is not None:
            bottleneck = self.prgate(bottleneck)

        logits, aux_outputs = self.decoder(
            bottleneck, [s5, s4, s3, s2_gated, s1]
        )

        uncertainty = self.evcal(logits) if self.evcal is not None else None
        return logits, aux_outputs, uncertainty


def build_model(config: dict, disabled_modules: Optional[List[str]] = None) -> BrainTumorSegModel:
    """Build a model from a config dict (see ``configs/config.yaml``)."""
    model_config = config.get("model", {})
    return BrainTumorSegModel(
        in_channels=config["data"]["in_channels"],
        num_classes=config["data"]["num_classes"],
        enc_features=model_config.get("enc_features", [32, 64, 128, 256, 320]),
        bottleneck_channels=model_config.get("bottleneck_channels", 320),
        dilation_rates=model_config.get("dilation_rates", [1, 2, 3]),
        coord_hidden=model_config.get("coord_hidden", 16),
        disabled_modules=disabled_modules,
        gating_variant=model_config.get("gating_variant", "coord_feature"),
    )
