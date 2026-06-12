from einops import rearrange
import torch
from torch.nn import functional as F
from torch import nn




class FFM_AdaptiveConcat2(nn.Module):
    def __init__(self, dimension=1, Channel1=32, Channel2=32, reduction=8):
        super().__init__()
        self.d = dimension
        self.C1 = Channel1
        self.C2 = Channel2
        self.C_all = Channel1 + Channel2

        self.weight_gen = nn.Sequential(
            nn.Linear(self.C_all * 2, max(8, self.C_all // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, self.C_all // reduction), self.C_all),
            nn.Sigmoid()
        )
        self.epsilon = 1e-6

    def forward(self, x1, x2):
        """
        x1: [B, C1, H, W]
        x2: [B, C2, H, W]
        """

        B, C1, _, _ = x1.size()
        _, C2, _, _ = x2.size()

        # -------- Global statistics --------
        mean1 = x1.mean(dim=(2, 3))
        std1  = x1.std(dim=(2, 3))

        mean2 = x2.mean(dim=(2, 3))
        std2  = x2.std(dim=(2, 3))

        stat = torch.cat([mean1, std1, mean2, std2], dim=1)

        # -------- Adaptive weights --------
        weight = self.weight_gen(stat)
        weight = weight / (weight.sum(dim=1, keepdim=True) + self.epsilon)

        w1, w2 = weight[:, :C1], weight[:, C1:]

        x1_w = x1 * w1.unsqueeze(-1).unsqueeze(-1)
        x2_w = x2 * w2.unsqueeze(-1).unsqueeze(-1)

        return torch.cat([x1_w, x2_w], dim=self.d)



