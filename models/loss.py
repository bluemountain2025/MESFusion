from typing import Literal
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from pytorch_msssim import MS_SSIM
from models.filters import BilateralFilter, Gradient
from models.models2 import FuseInput, FuseOutput




class ALL_Loss(nn.Module):
    def __init__(self,b,a, c: float, d: float):
        super().__init__()
        self.b =b
        self.a =a
        self.c = c
        self.d = d

        self.bila = BilateralFilter()
        self.grad = Gradient()
        self.ssim = MS_SSIM(data_range=1.0, channel=1)
        # self.downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)

    def intensity_loss(self, vi: Tensor, ir: Tensor, fu: Tensor):
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)

    # Put more emphasis on bright regions (hotspots)
        return F.mse_loss(fu, (vi + ir) / 2) + 0.5 * F.l1_loss(fu, torch.max(vi, ir))

    def grad_loss(self, vi, ir, fu):
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        vi_grad = self.grad(vi)
        ir_grad = self.grad(ir)
        # Use the maximum gradient magnitude of the two inputs as the target, not the sum of squares
        target_grad = torch.max(torch.abs(vi_grad), torch.abs(ir_grad))
        return F.mse_loss(self.grad(fu), target_grad)

    def ssim_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        return 1 - 0.5 * (self.ssim(fu, vi) + self.ssim(fu, ir))
    
    def tv_loss(self, fu: Tensor):
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        dx = torch.mean(torch.abs(fu[:, :, :, :-1] - fu[:, :, :, 1:]))
        dy = torch.mean(torch.abs(fu[:, :, :-1, :] - fu[:, :, 1:, :]))
        return dx + dy
    def fu_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)

        # l_contrast = self.contrast_loss(vi, ir, fu)
        l_int = self.intensity_loss(vi, ir, fu)
        l_ssim = self.ssim_loss(vi, ir, fu)
        l_grad = self.grad_loss(vi, ir, fu)
        l_tv = self.tv_loss(fu)
        return self.c * l_grad + self.d * l_ssim+self.b*l_int+self.a*l_tv

    


    def forward(self, inputs: FuseInput, outputs: FuseOutput) -> Tensor:
        # vi = self.downsample(inputs.vi)
        # ir = self.downsample(inputs.ir)
        
        vi = inputs.vi / 2 + 0.5
        ir = inputs.ir / 2 + 0.5

        fu = outputs.fusion / 2 + 0.5
        # fu = rgb_to_grayscale(fu)



        loss_fuse = self.fu_loss(vi, ir, fu)
        return loss_fuse

