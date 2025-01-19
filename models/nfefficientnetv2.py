import math
import numpy as np
import torch
import torch.nn as nn

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
else:
    from models.modules.activations import Swish

def _round_channels(c, divisor=8, min_value=None):
    """ Round number of channels to the nearest multiple of divisor."""
    if min_value is None:
        min_value = divisor
    
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor

    return new_c

def _round_repeats(repeats):
    """ Round number of repeats to the nearest integer."""
    return int(math.ceil(repeats))

def _drop_path(x, drop_prob, training):
    # """ Apply drop path regularization to input `x` during training."""
    # if drop_prob > 0 and training:
    #     keep_prob = 1 - drop_prob
    #     # Create the mask with the appropriate device and data type
    #     mask = torch.tensor(keep_prob, device=x.device, dtype=x.dtype).bernoulli_().expand_as(x)
    #     x = x / keep_prob  # Scale to maintain expected values
    #     x = x * mask
    return x

def mp_sum(a, b, weight=0.5):
    """ Take a weighted average between two tensors and scales the output to have consistent magnitude regardless of the weights.
    examples:
        weight = 0.5, both tensors are weighted equally and the output is scaled by 1/sqrt(2)
        weight = 0.0, only tensor a is used
        weight = 1.0, only tensor b is used
        """
    return torch.lerp(a, b, weight) / np.sqrt((1-weight)**2 + weight**2)

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

class MPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def normalize_weights(self, x, dim=None, eps=1e-4):
        if dim is None:
            dim = list(range(1, x.ndim))
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
        return x / norm.to(x.dtype)
    
    def standardize_weights(self, x, dim=None, eps=1e-4):
        # according to classic weight standardization
        if dim is None:
            dim = list(range(1, x.ndim))
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)
    
    def forward(self, x, gain=1):
        weight = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(self.normalize_weights(weight)) # forced weight normalization
        weight = self.normalize_weights(weight) # traditional weight normalization

        weight = weight * (gain / np.sqrt(weight[0].numel())) # magnitude-preserving scaling with the gain
        weight = weight.to(x.dtype)

        assert weight.ndim == 4
        return nn.functional.conv2d(x, 
                                    weight=self.weight, 
                                    bias=self.bias, 
                                    stride=self.stride, 
                                    padding=self.padding,
                                    dilation=self.dilation, 
                                    groups=self.groups)

class MPAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(MPAdaptiveAvgPool2d, self).__init__(output_size)
        raise NotImplementedError
    
    
class MPSqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio, activation=Swish, signal_preserving=False):
        super(MPSqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')
        squeeze_channels = int(squeeze_channels)

        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1)) # NOTE: will change mean and std
        self.se_reduce = MPConv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.se_act = activation(signal_preserving=signal_preserving)
        self.se_expand = MPConv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.se_act_scale = nn.Sigmoid()

    def forward(self, x):
        out = self.se_avgpool(x)
        out = self.se_act(self.se_reduce(out))
        scale = self.se_act_scale(self.se_expand(out)) # Place attention on every channel
        return x * scale

class Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 expand_ratio, 
                 se_ratio, 
                 drop_connect_rate, 
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(Block, self).__init__()

        expand_channels = in_channels * expand_ratio
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        conv = []

        # Point wise convolution phase
        if expand_ratio != 1:
            pointwise_conv1_layers = [MPConv2d(in_channels, expand_channels, 1, 1, 0, bias=False)]
            if normalization:
                pointwise_conv1_layers.append(nn.BatchNorm2d(expand_channels))
            pointwise_conv1_layers.append(activation(signal_preserving=signal_preserving))
            pointwise_conv1 = nn.Sequential(*pointwise_conv1_layers)

            conv.append(pointwise_conv1)

        # Depth wise convolution phase
        depthwise_conv_layers = [MPConv2d(expand_channels,
                                           expand_channels,
                                           kernel_size,
                                           stride,
                                           kernel_size // 2,
                                           groups=expand_channels,
                                           bias=False)]
        if normalization:
            depthwise_conv_layers.append(nn.BatchNorm2d(expand_channels))
        depthwise_conv_layers.append(activation(signal_preserving=signal_preserving))
        depthwise_conv = nn.Sequential(*depthwise_conv_layers)

        conv.append(depthwise_conv)

        # Squeeze and excite phase
        if se_ratio != 0:
            squeeze_excite = MPSqueezeAndExcite(expand_channels, in_channels, se_ratio, activation=activation, signal_preserving=signal_preserving)
            conv.append(squeeze_excite)

        # Projection phase
        pointwise_conv2_layers = [MPConv2d(expand_channels, out_channels, 1, 1, 0, bias=False)]
        if normalization:
            pointwise_conv2_layers.append(nn.BatchNorm2d(out_channels))
        pointwise_conv2 = nn.Sequential(*pointwise_conv2_layers)

        conv.append(pointwise_conv2)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            main_path = _drop_path(self.conv(x), self.drop_connect_rate, self.training)
            return mp_sum(x, main_path)
        else:
            return self.conv(x)

class NFEfficientNetv2(nn.Module):
    def __init__(self, 
                 model_variant="b0", 
                 num_classes=100, 
                 stem_channels=32, 
                 feature_size=1280, 
                 drop_connect_rate=0.2,
                 activation=Swish, # Pass the class, not the instance
                 signal_preserving=False,
                 normalization=True):
        super(NFEfficientNetv2, self).__init__()
        variants = {
            'b0': (1.0, 1.0, 224, 0.2),
            'b1': (1.0, 1.1, 240, 0.2),
            'b2': (1.1, 1.2, 260, 0.3),
            'b3': (1.2, 1.4, 300, 0.3),
            'b4': (1.4, 1.8, 380, 0.4),
            'b5': (1.6, 2.2, 456, 0.4),
            'b6': (1.8, 2.6, 528, 0.5),
            'b7': (2.0, 3.1, 600, 0.5)
        }
        config = [
            #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
            [32,  16,  3, 1, 1, 0.25, 1],
            [16,  24,  3, 2, 6, 0.25, 2],
            [24,  40,  5, 2, 6, 0.25, 2],
            [40,  80,  3, 2, 6, 0.25, 3],
            [80,  112, 5, 1, 6, 0.25, 3],
            [112, 192, 5, 2, 6, 0.25, 4],
            [192, 320, 3, 1, 6, 0.25, 1]
        ]

        # scaling width
        width_coefficient = variants[model_variant][0]
        if width_coefficient != 1:
            stem_channels = _round_channels(stem_channels*width_coefficient)
            for conf in config:
                conf[0] = _round_channels(conf[0]*width_coefficient)
                conf[1] = _round_channels(conf[1]*width_coefficient)

        # scaling depth
        depth_coefficient = variants[model_variant][1]
        if depth_coefficient != 1:
            for conf in config:
                conf[6] = _round_repeats(conf[6]*depth_coefficient)

        # scaling resolution
        # input_size = variants[model_variant][2]

        # stem convolution
        self.stem = [MPConv2d(in_channels=3, 
                               out_channels=stem_channels, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               bias=False)]
        if normalization:
            self.stem.append(nn.BatchNorm2d(stem_channels))
        self.stem.append(activation(signal_preserving=signal_preserving))
        self.stem = nn.Sequential(*self.stem)

        # construct the blocks based on the configuration of the specific model variant
        total_blocks = sum(conf[6] for conf in config)
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(Block(in_channels, 
                                      out_channels, 
                                      kernel_size, 
                                      stride, 
                                      expand_ratio, 
                                      se_ratio, 
                                      drop_rate, 
                                      activation=activation, 
                                      signal_preserving=signal_preserving, 
                                      normalization=normalization))
            
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(Block(out_channels, 
                                          out_channels, 
                                          kernel_size, 
                                          1, 
                                          expand_ratio, 
                                          se_ratio, 
                                          drop_rate, 
                                          activation=activation,
                                          signal_preserving=signal_preserving,
                                          normalization=normalization))
        self.blocks = nn.Sequential(*blocks)

        # head convolution
        self.head = [MPConv2d(in_channels=config[-1][1], 
                               out_channels=feature_size, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0, 
                               bias=False)]
        if normalization:
            self.head.append(nn.BatchNorm2d(feature_size))
        self.head.append(activation(signal_preserving=signal_preserving))
        self.head = nn.Sequential(*self.head)

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Flatten(),
            nn.Dropout(variants[model_variant][3]),
            nn.Conv2d(feature_size, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
 
        self._initialize_weights()

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = self.head(out)
        out = self.classifier(out)
        out = out.squeeze() # add in case you don't use a linear layer in the classifier
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MPConv2d): # you can also use nn.init.something here
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()

# Example usage
if __name__ == '__main__':
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from torchviz import make_dot

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    activation_class = load_activation_class('modules.activations', 'SELU')
    model = NFEfficientNetv2(model_variant="b0", 
                           num_classes=100, 
                           stem_channels=32, 
                           feature_size=1280, 
                           drop_connect_rate=0.2,
                           activation=activation_class,
                           signal_preserving=False,
                           normalization=True)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda")
    target = torch.randint(1, 100, (10,)).to("cuda")
    x = torch.randn(10, 3, 32, 32).to("cuda")
    output = model(x)
    loss = criterion(output, target)
    print(loss)
    print(output.shape)

    # Visualize model
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())