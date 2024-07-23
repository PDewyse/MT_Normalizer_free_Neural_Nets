import torch
import torch.nn as nn
from activations import Swish

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            Swish()
        )

    def forward(self, x):
        return self.depthwise_conv(x)

# Test
model = DepthwiseConv(in_channels=32, kernel_size=3, stride=1)
input_tensor = torch.rand(1, 32, 32, 32)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 32, 32, 32])
