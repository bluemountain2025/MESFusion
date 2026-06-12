from thop import profile, clever_format
import torch
from models.models import FuseInput, DBTFuse1, FuseOutput

if __name__ == '__main__':
    model = DBTFuse1(
        in_channels=1,
        out_channels=1,
        num_features=32,

        patch_size=16,

     
    
    ).cuda().eval()

    vi=torch.randn(1, 1, 256, 256).cuda()
    ir=torch.randn(1, 1, 256, 256).cuda()
    # out: FuseOutput = model(FuseInput(vi, ir))
    # print('fusion', out.fusion.shape)
    # print('ir_reconstruction', out.ir_reconstruction.shape)
    # print('vi_reconstruction', out.vi_reconstruction.shape)
    # print('ir_feature_private', out.ir_feature_private.shape)
    # print('vi_feature_private', out.vi_feature_private.shape)
    # print('ir_feature_common', out.ir_feature_common.shape)
    # print('vi_feature_commom', out.vi_feature_commom.shape)

    from thop import profile, clever_format
    FLOPs, Params = profile(model, inputs=(FuseInput(vi, ir),))
    print(FLOPs, Params)
    FLOPs, Params = clever_format([FLOPs * 2, Params], '%.4f')
    print(f'{FLOPs=}, {Params=}')