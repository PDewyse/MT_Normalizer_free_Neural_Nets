import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: maybe take the pytorch implementation instead
# eg: self.spectral_norm = nn.utils.spectral_norm
class SpectralNorm(nn.Module):
    """ Spectral Normalization for Weight Parameters 
    Args:
        module: nn.Module. containing weight parameters to be normalized
        name: str. name of weight parameters
        power_iterations: int. number of iterations to update the weight
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
        
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w.view(height, -1).data, v.data), dim=0, eps=1e-12)
        
        sigma = torch.dot(u, torch.mv(w.view(height, -1).data, v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = F.normalize(w.data.new(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(w.data.new(width).normal_(0, 1), dim=0, eps=1e-12)
        w_bar = w.data

        self.module.register_parameter(self.name + "_u", nn.Parameter(u, requires_grad=False))
        self.module.register_parameter(self.name + "_v", nn.Parameter(v, requires_grad=False))
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w_bar, requires_grad=True))

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# example usages
class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, power_iterations=1):
        super(SpectralNormConv2d, self).__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv = SpectralNorm(conv, power_iterations=power_iterations)

    def forward(self, x):
        return self.conv(x)

# Example of using SpectralNormConv2d
spectral_norm_conv = SpectralNormConv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Testing the layer with a random input
input_tensor = torch.randn(1, 3, 32, 32)
output_tensor = spectral_norm_conv(input_tensor)
print(output_tensor.shape)
