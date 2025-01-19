import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
else:
    from models.modules.activations import Swish

def sp_sum(a, b, weight=0.5):
    """ Signal preserving sum of two tensors."""
    return torch.lerp(a, b, weight) / np.sqrt((1-weight)**2 + weight**2)

# use alpha drop out
def _drop_path(x, drop_prob, training):
    """ Apply alpha dropout regularization to input `x` during training."""
    if drop_prob > 0 and training:
        x = F.alpha_dropout(x, p=drop_prob, training=training)
    return x

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio, activation=Swish, signal_preserving=False):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')
        squeeze_channels = int(squeeze_channels)

        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.se_act = activation(signal_preserving=signal_preserving)
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.se_act_scale = nn.Sigmoid()

    def forward(self, x):
        out = self.se_avgpool(x)
        out = self.se_act(self.se_reduce(out))
        out = self.se_act_scale(self.se_expand(out))
        return x * out

class SEBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, stride, 
                 se_ratio, 
                 drop_connect_rate, 
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(SEBlock, self).__init__()

        self.drop_connect_rate = drop_connect_rate
        conv = []

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            *([nn.BatchNorm2d(out_channels)] if normalization else []),
            activation(signal_preserving=signal_preserving))

        self.convblock2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            *([nn.BatchNorm2d(out_channels)] if normalization else []),
            activation(signal_preserving=signal_preserving))
        
        self.se = SqueezeAndExcite(out_channels, in_channels, se_ratio, activation=activation, signal_preserving=signal_preserving)

        conv = [self.convblock1, self.convblock2, self.se]

        # Residual connection if input and output channels match
        self.norm = normalization
        self.residual_connection = (in_channels == out_channels) and (stride == 1)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            path = _drop_path(self.conv(x), self.drop_connect_rate, self.training)
            return x + path if self.norm else sp_sum(x, path)
        else:
            return self.conv(x)

class SENet(nn.Module):
    def __init__(self, 
                 num_classes=100, 
                 stem_channels=32, 
                 feature_size=1280, 
                 drop_connect_rate=0.2, 
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True): # so you can easily turn BatchNorm off
        super(SENet, self).__init__()
        
        # stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            *([nn.BatchNorm2d(stem_channels)] if normalization else []),
            activation(signal_preserving=signal_preserving))

        # SE blocks
        self.blocks = nn.Sequential(
            SEBlock(stem_channels, 32, 3, 1, 0.25, drop_connect_rate, activation=activation, signal_preserving=signal_preserving, normalization=normalization),
            SEBlock(32, 64, 5, 1, 0.25, drop_connect_rate, activation=activation, signal_preserving=signal_preserving, normalization=normalization),
            SEBlock(64, 64, 3, 2, 0.25, drop_connect_rate, activation=activation, signal_preserving=signal_preserving, normalization=normalization),
            SEBlock(64, 256, 5, 1, 0.25, drop_connect_rate, activation=activation, signal_preserving=signal_preserving, normalization=normalization)
        )

        # head convolution
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=feature_size, kernel_size=1, stride=1, padding=0, bias=False),
            *([nn.BatchNorm2d(feature_size)] if normalization else []),
            activation(signal_preserving=signal_preserving))

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_size, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                m.bias.data.zero_()

# Example usage
if __name__ == '__main__':
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from torchviz import make_dot

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    activation_class = load_activation_class('modules.activations', 'Swish')
    model = SENet(num_classes=100, 
                         stem_channels=32, 
                         feature_size=1280, 
                         drop_connect_rate=0.2, 
                         activation=activation_class,
                         signal_preserving=False,
                         normalization=False)
    

    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda")
    target = torch.randint(1, 100, (10,)).to("cuda")
    x = torch.randn(10, 3, 32, 32).to("cuda")
    output = model(x)
    loss = criterion(output, target)
    print(loss)
    print(output.shape)

    # Visualize model
    print(model)
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render(filename=model.__class__.__name__, directory='models/visualizations', format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())