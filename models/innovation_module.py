import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gradient3D(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        center = kernel_size // 2
        x = torch.arange(kernel_size).float() - center
        y = torch.arange(kernel_size).float() - center
        z = torch.arange(kernel_size).float() - center
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
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
        x_flat = x.view(b * c, 1, d, h, w)
        pad = self.gx.shape[2] // 2
        grad_x = F.conv3d(x_flat, self.gx, padding=pad)
        grad_y = F.conv3d(x_flat, self.gy, padding=pad)
        grad_z = F.conv3d(x_flat, self.gz, padding=pad)
        grad = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2).sqrt()
        grad = grad.view(b, c, d, h, w)
        return grad.max(dim=1, keepdim=True).values


class BoundGATE(nn.Module):
    """BoundGATE: Multi-scale boundary gradient → Sigmoid gating for Stage 2 features.
    Type: T3 | Ladder: L3
    Eq.1: s2_gated = s2 ⊙ σ(Conv(cat[s2, g1↑, g2]))"""

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.grad1 = Gradient3D(sigma=1.0)
        self.grad2 = Gradient3D(sigma=2.0)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels + 2, in_channels, 1),
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.Sigmoid(),
        )

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        g1 = self.grad1(s1)
        g2 = self.grad2(s2)
        g1_up = F.interpolate(g1, size=s2.shape[2:], mode="trilinear", align_corners=False)
        gate = self.gate_conv(torch.cat([s2, g1_up, g2], dim=1))
        return s2 * gate


class PyraGATE(nn.Module):
    """PyraGATE: Voxel coords → MLP → α gated tri-dilation DWConv (d=1,2,3) fusion.
    Type: T3 | Ladder: L3→L4 (core contribution)
    Eq.2: F' = Σ α_i · DWConv_{d=i}(F), α = softmax(MLP(d,h,w))"""

    def __init__(self, channels: int = 256, dilation_rates: list = None,
                 coord_hidden: int = 16):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 3]
        self.dilation_rates = dilation_rates
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=d, dilation=d, groups=channels),
                nn.Conv3d(channels, channels, 1),
                nn.LeakyReLU(0.01, inplace=True),
            )
            for d in dilation_rates
        ])
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, coord_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(coord_hidden, len(dilation_rates)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        gd = torch.linspace(0, 1, d, device=x.device)
        gh = torch.linspace(0, 1, h, device=x.device)
        gw = torch.linspace(0, 1, w, device=x.device)
        grid_d, grid_h, grid_w = torch.meshgrid(gd, gh, gw, indexing='ij')
        coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)
        alpha = self.coord_mlp(coords)
        alpha = alpha.permute(3, 0, 1, 2).unsqueeze(0)
        alpha = F.softmax(alpha, dim=1)

        out = 0
        for i, conv in enumerate(self.convs):
            f = conv(x)
            out = out + alpha[:, i:i+1] * f
        return x + out


class EKDCalib(nn.Module):
    """EKD-CALIB: Calibrated evidential uncertainty with learnable temperature τ.
    Type: T3 | Ladder: L3
    Eq.3: evidence = softmax(z/τ), α = e+1, U = K/Σα"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.log_tau = nn.Parameter(torch.zeros(1))

    @property
    def tau(self):
        return torch.exp(self.log_tau).clamp(min=0.01, max=10.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        tau = self.tau
        evidence = F.softmax(logits / tau, dim=1)
        alpha = evidence + 1.0
        total_evidence = alpha.sum(dim=1, keepdim=True)
        uncertainty = self.num_classes / total_evidence
        return uncertainty.clamp(min=0.5, max=1.0)

    def get_u_weight(self, logits: torch.Tensor) -> torch.Tensor:
        uncertainty = self.forward(logits)
        return (1.0 + uncertainty.detach()).mean()
