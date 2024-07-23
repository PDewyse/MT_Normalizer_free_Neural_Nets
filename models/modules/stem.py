import torch
import torch.nn as nn
from activations import Swish

class Stem(nn.Module):
    def __init__(self, stem_channels=32):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            Swish()
        )

    def forward(self, x):
        return self.stem(x)

# Test
model = Stem(stem_channels=32)
input_tensor = torch.rand(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)  # Expected: torch.Size([1, 32, 112, 112])
