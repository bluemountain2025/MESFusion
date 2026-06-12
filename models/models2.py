from dataclasses import dataclass
# from itertools import pairwise
# from torch.nn import LayerNorm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.CMblock import FreqMambaLite
from models.Edge import Func_PreEnhance, MRI_PreEnhance
from models.fusionlayer import FFM_AdaptiveConcat2
@dataclass
class FuseInput:
    vi: Tensor
    ir: Tensor

@dataclass
class FuseOutput:
    fusion: Tensor
    ir_out: Tensor
    vi_out: Tensor


# Decoder
class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: list[int]):
        super().__init__()
        # We now need 3 intermediate channels, so length == 3
        assert len(mid_channels) == 3, "mid_channels list must contain exactly three elements."

        # Layer 1: in_channels -> mid1
        self.conv1 = nn.Conv2d(in_channels,   mid_channels[0], kernel_size=3, padding=1)
        self.relu1 = nn.SiLU(inplace=True)

        # Layer 2: mid1 -> mid2
        self.conv2 = nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: mid2 -> mid3
        self.conv3 = nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Layer 4: mid3 -> 1
        self.conv4 = nn.Conv2d(mid_channels[2], 1,             kernel_size=3, padding=1)
        self.tanh  = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # B, mid1, H, W
        x = self.relu2(self.conv2(x))  # B, mid2, H, W
        x = self.relu3(self.conv3(x))  # B, mid3, H, W
        x = self.tanh (self.conv4(x))  # B,    1, H, W
        return x



    
# Overall model
class DBTFuse1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,

        patch_size: 16
    ):
        super().__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fun_enchan=Func_PreEnhance(channels=16)
        self.MRI_enchan=MRI_PreEnhance(channels=16)
        self.Mam = FreqMambaLite(d_model=16)
        self.conv_ir = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        decoder_channels = [num_features + num_features, 16, 8,4] # Channel sizes for the decoder layers
        self.decoder_vi = SimpleDecoder(32, decoder_channels[1:])
        self.decoder_ir = SimpleDecoder(32, decoder_channels[1:])
        self.decoder_fuse = SimpleDecoder(64, decoder_channels[out_channels:])
        self.fusion_strategy = FFM_AdaptiveConcat2()
        # print("kernel", (self.encode_ir.out_channels))

        self.proj_out = nn.Tanh()


    def forward(self, inputs: FuseInput):

        ir2 = self.conv_ir(inputs.ir)  # Project inputs.ir from 1 channel to 16 channels

        vi2 = self.conv_ir(inputs.vi)  # Project inputs.vi from 1 channel to 16 channels
        ir_en=self.MRI_enchan(ir2) # ir denotes MRI
        vi_en=self.fun_enchan(vi2) # vi denotes all other modalities
        vi_M = self.Mam(ir_en)
        ir_M = self.Mam(vi_en)
        # print("ir",ir_M.shape)
        # print("vi",vi_M.shape)
        merged = self.fusion_strategy(vi_M,ir_M)

        r_ir = self.proj_out(self.decoder_ir(ir_M))
        r_vi = self.proj_out(self.decoder_vi(vi_M))
        fusion = self.proj_out(self.decoder_fuse(merged))
       
        return FuseOutput(
            fusion=fusion.clamp(-1.0, 1.0),
            vi_out=r_vi.clamp(-1.0, 1.0),
            ir_out=r_ir.clamp(-1.0, 1.0)
        )

    def frozen(self, *__names: str):
        for name, param in self.named_parameters():
            for _name in __names:
                if _name in name:
                    param.requires_grad = False
                    print('frozen layer:', name)

if __name__ == '__main__':
    model = DBTFuse1(
        in_channels=1,
        out_channels=1,
        num_features=48,
        patch_size= 16

    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vi = torch.randn(1,1, 224, 224).to(device)  # input VIS,batchsize=3,channel=3
    ir = torch.randn(1, 1,224, 224).to(device)  # input IR

    fuse_input = FuseInput(vi=vi, ir=ir)

    with torch.no_grad():  
        fuse_output = model(fuse_input)


    print(f"Fusion output shape: {fuse_output.fusion.shape}")
