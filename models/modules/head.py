import torch
import torch.nn as nn
from models.modules.activations import Swish

class Head(nn.Module):
    def __init__(self, in_channels=320, feature_size=1280):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, 1, bias=False),
            nn.BatchNorm2d(feature_size),
            Swish()
        )

    def forward(self, x):
        return self.head(x)

# Test
model = Head(in_channels=320, feature_size=1280)
input_tensor = torch.rand(1, 320, 7, 7)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 1280, 7, 7])
