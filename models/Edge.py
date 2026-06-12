import torch
import torch.nn as nn
import torch.nn.functional as F

## Process MRI images
class MRI_PreEnhance(nn.Module):
    """
    MRI Pre-Enhancement Module (MPE)
    - Structure-aware enhancement
    - Adaptive contrast normalization
    Input:  MRI image / feature map  [B, C, H, W]
    Output: Enhanced MRI feature     [B, C, H, W]
    """

    def __init__(self, channels, reduction=8, eps=1e-5):
        super().__init__()
        self.eps = eps

        # -------- Multi-scale gradient extraction --------
        self.grad_3x3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.grad_5x5 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2,
            groups=channels, bias=False
        )

        # -------- Structure attention --------
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # -------- Adaptive contrast parameters --------
        self.gamma = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: [B, C, H, W]

        # ----- gradient / structure features -----
        g3 = torch.abs(self.grad_3x3(x))
        g5 = torch.abs(self.grad_5x5(x))
        g = g3 + g5

        # ----- structure-aware attention -----
        attn_input = torch.cat([x, g3, g5], dim=1)
        A_str = self.attention(attn_input)
        x_str = x + A_str * g

        # ----- adaptive contrast normalization -----
        mean = x_str.mean(dim=(2, 3), keepdim=True)
        std = x_str.std(dim=(2, 3), keepdim=True)

        x_norm = (x_str - mean) / (std + self.eps)
        x_out = self.gamma(x_norm) + self.beta(x_norm)

        return x_out

## Process functional images such as SPECT, CT, and PET
class Func_PreEnhance(nn.Module):
    """
    Functional / Density Pre-Enhancement Module (FPE)
    - Intensity statistics modeling
    - Energy-aware attention
    - Dynamic range re-scaling
    Input:  PET / SPECT / CT image or feature [B, C, H, W]
    Output: Enhanced functional feature       [B, C, H, W]
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        # -------- Intensity statistics encoding --------
        self.stat_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # -------- Dynamic range parameters --------
        self.alpha = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: [B, C, H, W]

        # ----- intensity statistics -----
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)

        stat = torch.cat([mean, std], dim=1)
        A_eng = self.stat_encoder(stat)

        # ----- energy-aware enhancement -----
        x_eng = x * A_eng

        # ----- dynamic range re-scaling -----
        x_centered = x_eng - mean
        x_scaled = torch.tanh(self.alpha(x_centered)) + self.beta(x_centered)

        return x_scaled
