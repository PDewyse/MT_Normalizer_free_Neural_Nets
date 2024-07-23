import torch
import torch.nn as nn
from activations import Swish

class PointwiseDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(PointwiseDepthwiseConv, self).__init__()
        self.expand_channels = in_channels * 6
        
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.expand_channels),
            Swish()
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.expand_channels, self.expand_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.expand_channels, bias=False),
            nn.BatchNorm2d(self.expand_channels),
            Swish()
        )
        
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv2d(self.expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv2(out)
        return out

# Test
model = PointwiseDepthwiseConv(in_channels=32, out_channels=64, kernel_size=3, stride=1)
input_tensor = torch.rand(1, 32, 32, 32)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 64, 32, 32])
