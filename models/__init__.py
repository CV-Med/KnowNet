from typing import List, Optional
import torch
import torch.nn as nn
from models.backbone import Encoder, Decoder
from models.innovation_module import BoundGATE, PyraGATE, EKDCalib


class BrainTumorSegModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4,
                 enc_features: list = None, bottleneck_channels: int = 320,
                 dilation_rates: list = None, coord_hidden: int = 16,
                 disabled_modules: Optional[List[str]] = None):
        super().__init__()
        if enc_features is None:
            enc_features = [32, 64, 128, 256, 320]
        if dilation_rates is None:
            dilation_rates = [1, 2, 3]
        if disabled_modules is None:
            disabled_modules = []

        self.disabled_modules = disabled_modules

        self.encoder = Encoder(in_channels, enc_features)
        self.bottleneck = nn.Sequential(
            nn.Conv3d(enc_features[-1], bottleneck_channels, 3, padding=1),
            nn.GroupNorm(min(8, bottleneck_channels), bottleneck_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.decoder = Decoder(enc_features, bottleneck_channels, num_classes)

        if "BoundGATE" not in disabled_modules:
            self.boundgate = BoundGATE(in_channels=enc_features[1])
        else:
            self.boundgate = None

        if "PyraGATE" not in disabled_modules:
            self.pyragate = PyraGATE(bottleneck_channels, dilation_rates, coord_hidden)
        else:
            self.pyragate = None

        if "EKD_CALIB" not in disabled_modules:
            self.ekd_calib = EKDCalib(num_classes=num_classes)
        else:
            self.ekd_calib = None

    def forward(self, x: torch.Tensor):
        s1, s2, s3, s4, s5, p5 = self.encoder(x)

        if self.boundgate is not None:
            s2_gated = self.boundgate(s1, s2)
        else:
            s2_gated = s2

        bottleneck = self.bottleneck(p5)

        if self.pyragate is not None:
            bottleneck = self.pyragate(bottleneck)

        logits, aux_outputs = self.decoder(bottleneck, [s5, s4, s3, s2_gated, s1])

        if self.ekd_calib is not None:
            uncertainty = self.ekd_calib(logits)
        else:
            uncertainty = None

        return logits, aux_outputs, uncertainty


def build_model(config: dict, disabled_modules: Optional[List[str]] = None) -> BrainTumorSegModel:
    model_config = config.get("model", {})
    model = BrainTumorSegModel(
        in_channels=config["data"]["in_channels"],
        num_classes=config["data"]["num_classes"],
        enc_features=model_config.get("enc_features", [32, 64, 128, 256]),
        bottleneck_channels=model_config.get("bottleneck_channels", 256),
        dilation_rates=model_config.get("dilation_rates", [1, 2, 3]),
        coord_hidden=model_config.get("coord_hidden", 16),
        disabled_modules=disabled_modules,
    )
    return model
