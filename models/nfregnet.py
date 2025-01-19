import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps  # Epsilon, a small constant to avoid dividing by zero.

    def get_weight(self):
        # Get Scaled WS weight
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

class SqueezeExcite(nn.Module):
    """Simple Squeeze+Excite layers."""
    def __init__(self, in_channels, width, activation):
        super().__init__()
        self.se_conv0 = nn.Conv2d(in_channels, width, kernel_size=1, bias=True)
        self.se_conv1 = nn.Conv2d(width, in_channels, kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x):
        h = torch.mean(x, axis=[2, 3], keepdims=True)  # Mean pool for NCHW tensors
        h = self.se_conv1(self.activation(self.se_conv0(h)))  # Apply two linear layers with activation in between
        return (torch.sigmoid(h) * 2) * x  # Rescale the sigmoid output and return

class NFBlock(nn.Module):
    """NF-RegNet block."""
    def __init__(self, in_channels, out_channels, stride=1, activation=F.relu, which_conv=ScaledWSConv2d, beta=1.0, alpha=1.0, expansion=2.25, se_ratio=0.5, group_size=8):
        super().__init__()
        self.in_ch, self.out_ch = in_channels, out_channels
        self.activation = activation
        self.beta, self.alpha = beta, alpha
        width = int(in_channels * expansion)
        groups = width // group_size if group_size else 1
        width = int(group_size * groups)  # Round width up if you pick a bad group size
        self.stride = stride
        self.width = width

        self.conv1x1a = which_conv(in_channels, width, kernel_size=1, padding=0)
        self.conv3x3 = which_conv(width, width, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.conv1x1b = which_conv(width, out_channels, kernel_size=1, padding=0)
        self.conv_shortcut = which_conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if stride > 1 or in_channels != out_channels else None

        # Hidden size of the S+E MLP
        se_width = max(1, int(width * se_ratio))
        self.se = SqueezeExcite(width, se_width, self.activation)
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        out = self.activation(x) / self.beta
        shortcut = self.conv_shortcut(F.avg_pool2d(out, 2)) if self.conv_shortcut is not None else x
        out = self.conv1x1a(out)  # Initial bottleneck conv
        out = self.conv3x3(self.activation(out))  # Spatial conv
        out = self.se(out)  # Apply squeeze + excite to middle block.
        out = self.conv1x1b(self.activation(out))
        return out * self.skipinit_gain * self.alpha + shortcut

# Nonlinearities with baked constants
nonlinearities = {
    'silu': lambda x: F.silu(x) / .5595,
    'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
    'identity': lambda x: x
}

# Block base widths and depths for each variant
params = {
    'NF-RegNet-B0': {'width': [48, 104, 208, 440], 'depth': [1, 3, 6, 6], 'train_imsize': 192, 'test_imsize': 224, 'weight_decay': 2e-5, 'drop_rate': 0.2},
    'NF-RegNet-B1': {'width': [48, 104, 208, 440], 'depth': [2, 4, 7, 7], 'train_imsize': 240, 'test_imsize': 256, 'weight_decay': 2e-5, 'drop_rate': 0.2},
    # Add other variants as needed...
}

def count_params(module):
    return sum([item.numel() for item in module.parameters()])

class NFRegNet(nn.Module):
    """Normalizer-Free RegNets."""
    def __init__(self, variant='NF-RegNet-B0', num_classes=1000, width=0.75, expansion=2.25, se_ratio=0.5, group_size=8, alpha=0.2, activation='silu', drop_rate=None, stochdepth_rate=0.0):
        super().__init__()
        self.variant = variant
        self.width = width
        self.expansion = expansion
        self.num_classes = num_classes
        self.se_ratio = se_ratio
        self.group_size = group_size
        self.alpha = alpha
        self.activation = nonlinearities.get(activation)

        # Drop rate and stochdepth rate
        self.drop_rate = drop_rate if drop_rate is not None else params[self.variant]['drop_rate']
        self.stochdepth_rate = stochdepth_rate

        self.which_conv = functools.partial(ScaledWSConv2d, gain=True, bias=True)

        # Get width and depth pattern
        self.width_pattern = [int(val * self.width) for val in params[variant]['width']]
        self.depth_pattern = params[self.variant]['depth']

        # Stem conv
        in_channels = int(self.width_pattern[0])
        self.initial_conv = self.which_conv(3, in_channels, kernel_size=3, stride=2, padding=1)

        # Body
        blocks = []
        expected_var = 1.0
        index = 0
        for block_width, stage_depth in zip(self.width_pattern, self.depth_pattern):
            for block_index in range(stage_depth):
                expand_ratio = expansion if index > 0 else 1
                beta = expected_var ** 0.5
                blocks.append(NFBlock(in_channels, block_width, stride=2 if block_index == 0 else 1, activation=self.activation, which_conv=self.which_conv, beta=beta, alpha=self.alpha, expansion=expand_ratio, se_ratio=self.se_ratio, group_size=self.group_size))
                in_channels = block_width
                expected_var = 1. if block_index == 0 else expected_var + self.alpha ** 2
                index += 1
        self.blocks = nn.Sequential(*blocks)

        # Final convolution
        ch = int(1280 * in_channels // 440)
        self.final_conv = self.which_conv(in_channels, ch, kernel_size=1, padding=0)
        in_channels = ch

        if self.drop_rate > 0.0:
            self.dropout = nn.Dropout(drop_rate)

        # Classifier layer with zero-initialized weights
        self.fc = nn.Linear(in_channels, self.num_classes, bias=True)
        torch.nn.init.zeros_(self.fc.weight)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.blocks(out)
        out = self.activation(self.final_conv(out))
        pool = torch.mean(out, [2, 3])
        if self.drop_rate > 0.0 and self.training:
            pool = self.dropout(pool)
        return self.fc(pool)

# example usage
if __name__ == '__main__':
    from torchviz import make_dot
    model = NFRegNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(count_params(model))

    # Visualize model
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')
