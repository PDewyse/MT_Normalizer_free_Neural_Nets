import torch
import torch.nn as nn
from activations import Swish

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.pointwise_conv(x)

# Test
model = PointwiseConv(in_channels=32, out_channels=64)
input_tensor = torch.rand(1, 32, 32, 32)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 64, 32, 32])
