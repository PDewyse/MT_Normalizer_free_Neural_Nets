import torch
import torch.nn as nn
from activations import Swish
from squeezeandexcite import SqueezeAndExcite

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        expand_channels = in_channels * expand_ratio
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size, stride, kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish()
        )
        
        self.squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.squeeze_excite(out)
        out = self.project_conv(out)
        if self.residual_connection:
            return x + out
        else:
            return out

# Test
model = MBConvBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25)
input_tensor = torch.rand(1, 32, 32, 32)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 16, 32, 32])
