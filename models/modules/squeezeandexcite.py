import torch
import torch.nn as nn
from activations import Swish

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeAndExcite, self).__init__()
        squeeze_channels = max(1, int(in_channels * se_ratio))
        
        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_reduce = nn.Conv2d(in_channels, squeeze_channels, 1, bias=True)
        self.se_act1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, in_channels, 1, bias=True)
        self.se_act2 = nn.Sigmoid()

    def forward(self, x):
        out = self.se_avgpool(x)
        out = self.se_reduce(out)
        out = self.se_act1(out)
        out = self.se_expand(out)
        out = self.se_act2(out)
        return x * out

# Test
model = SqueezeAndExcite(in_channels=32)
input_tensor = torch.rand(1, 32, 32, 32)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 32, 32, 32])