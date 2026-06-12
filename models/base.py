import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

#DWconv2d
class DWConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        weights: Tensor | None = None,
        bias: Tensor | None = None,
        *,
        kernel_size: _size_2_t = 1,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        with_bias: bool = True,
        padding_mode: str = 'replicate',  # TODO: refine this type
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        kernel_size = weights.shape if weights is not None else kernel_size
        with_bias = True if bias is not None else with_bias
        self.dw = nn.Conv2d(channels, channels, kernel_size,
                        stride, padding, dilation, channels,
                        with_bias, padding_mode, device, dtype)
        
        if weights is not None:
            w = weights.repeat(channels, 1, 1, 1).type(torch.float32)
            self.dw.weight = nn.Parameter(w, requires_grad=False)
        if bias is not None:
            self.dw.bias = nn.Parameter(bias.type(torch.float32), requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dw(x)
    



# input_tensor = torch.rand(1, 1, 10, 10)  # Batch size is 1, channel count is 1, and image size is 10x10

# # Define the convolution kernel and bias
# kernel = torch.tensor([[1.0, 0.0, -1.0],
#                        [1.0, 0.0, -1.0],
#                        [1.0, 0.0, -1.0]])
# kernel = kernel.unsqueeze(0)  # Add an extra batch dimension

# # You can optionally define a bias; here we choose not to use one
# bias = None  # Or: bias = torch.tensor([1.0])

# # Create a DWConv2d instance
# conv_layer = DWConv2d(channels=1, weights=kernel, bias=bias, kernel_size=3, padding=1)
# print(input_tensor.shape)
# # Apply the convolution layer
# output = conv_layer(input_tensor)

# # Print the output
# print("Input:")
# print(input_tensor)
# print("Output:")
# print(output)
