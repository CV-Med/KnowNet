import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, groups: int = 8, eps: float = 1e-3):
    groups = min(groups, channels)
    return nn.GroupNorm(groups, channels, eps=eps)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                _group_norm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels: int, features: list):
        super().__init__()
        self.stage1 = ResBlock3D(in_channels, features[0])
        self.down1 = nn.Conv3d(features[0], features[0], 2, stride=2)
        self.stage2 = ResBlock3D(features[0], features[1])
        self.down2 = nn.Conv3d(features[1], features[1], 2, stride=2)
        self.stage3 = ResBlock3D(features[1], features[2])
        self.down3 = nn.Conv3d(features[2], features[2], 2, stride=2)
        self.stage4 = ResBlock3D(features[2], features[3])
        self.down4 = nn.Conv3d(features[3], features[3], 2, stride=2)
        self.stage5 = ResBlock3D(features[3], features[4])
        self.down5 = nn.Conv3d(features[4], features[4], 2, stride=2)

    def forward(self, x: torch.Tensor):
        s1 = self.stage1(x)
        p1 = self.down1(s1)
        s2 = self.stage2(p1)
        p2 = self.down2(s2)
        s3 = self.stage3(p2)
        p3 = self.down3(s3)
        s4 = self.stage4(p3)
        p4 = self.down4(s4)
        s5 = self.stage5(p4)
        p5 = self.down5(s5)
        return s1, s2, s3, s4, s5, p5


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = ResBlock3D(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, features: list, bottleneck_channels: int, num_classes: int):
        super().__init__()
        rev = features[::-1]
        self.up5 = DecoderBlock(bottleneck_channels, rev[0], rev[0])
        self.up4 = DecoderBlock(rev[0], rev[1], rev[1])
        self.up3 = DecoderBlock(rev[1], rev[2], rev[2])
        self.up2 = DecoderBlock(rev[2], rev[3], rev[3])
        self.up1 = DecoderBlock(rev[3], rev[4], rev[4])
        self.final_conv = nn.Conv3d(rev[4], num_classes, 1)
        self.aux_convs = nn.ModuleList([
            nn.Conv3d(rev[i], num_classes, 1) for i in range(1, len(rev))
        ])

    def forward(self, x: torch.Tensor, skips: list):
        aux_outputs = []
        d5 = self.up5(x, skips[0])
        d4 = self.up4(d5, skips[1])
        aux_outputs.append(self.aux_convs[0](d4))
        d3 = self.up3(d4, skips[2])
        aux_outputs.append(self.aux_convs[1](d3))
        d2 = self.up2(d3, skips[3])
        aux_outputs.append(self.aux_convs[2](d2))
        d1 = self.up1(d2, skips[4])
        aux_outputs.append(self.aux_convs[3](d1))
        logits = self.final_conv(d1)
        return logits, aux_outputs
