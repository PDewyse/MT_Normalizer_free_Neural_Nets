import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
    from modules.dynamictanh import DynamicTanh
else:
    from models.modules.activations import Swish
    from models.modules.dynamictanh import DynamicTanh

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

# use alpha drop out
def _drop_path(x, drop_prob, training):
    # """ Apply alpha dropout regularization to input `x` during training."""
    # if drop_prob > 0 and training:
    #     x = F.alpha_dropout(x, p=drop_prob, training=training)
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

class MBConvBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, stride, 
                 expand_ratio, se_ratio, 
                 drop_connect_rate, 
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True,
                 alpha=1):
        super(MBConvBlock, self).__init__()

        expand_channels = in_channels * expand_ratio
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        self.norm = normalization

        conv = []
        # Point wise convolution phase
        if expand_ratio != 1:
            pointwise_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                *([nn.BatchNorm2d(expand_channels)] if self.norm else [DynamicTanh(expand_channels, False, alpha)]),
                activation(signal_preserving=signal_preserving)
                )
            conv.append(pointwise_conv1)

        # Depth wise convolution phase
        depthwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size, stride, kernel_size // 2, groups=expand_channels, bias=False),
            *([nn.BatchNorm2d(expand_channels)] if self.norm else [DynamicTanh(expand_channels, False, alpha)]),
            activation(signal_preserving=signal_preserving)
            )
        conv.append(depthwise_conv)

        # Squeeze and excite phase
        if se_ratio != 0:
            conv.append(SqueezeAndExcite(expand_channels, in_channels, se_ratio, activation=activation, signal_preserving=signal_preserving))

        # Projection phase
        pointwise_conv2 = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            *([nn.BatchNorm2d(out_channels)] if self.norm else [DynamicTanh(out_channels, False, alpha)])
            )
        conv.append(pointwise_conv2)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            main_path = _drop_path(self.conv(x), self.drop_connect_rate, self.training)
            return x + main_path
        else:
            return self.conv(x)

class DyTSNEfficientNet(nn.Module):
    def __init__(self, 
                 model_variant="b0", 
                 num_classes=100, 
                 stem_channels=32, 
                 feature_size=1280, 
                 drop_connect_rate=0.2, 
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True,
                 alpha=1): # so you can easily turn BatchNorm off
        super(DyTSNEfficientNet, self).__init__()
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
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            *([nn.BatchNorm2d(stem_channels)] if normalization else [DynamicTanh(stem_channels, False, alpha)]),
            activation(signal_preserving=signal_preserving)
            )
        
        # mobile inverted bottleneck
        total_blocks = sum(conf[6] for conf in config)
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channels, 
                                      out_channels, 
                                      kernel_size, 
                                      stride, 
                                      expand_ratio, 
                                      se_ratio, 
                                      drop_rate, 
                                      activation=activation, 
                                      signal_preserving=signal_preserving, 
                                      normalization=normalization,
                                      alpha=alpha))
            
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channels, 
                                          out_channels, 
                                          kernel_size, 
                                          1, 
                                          expand_ratio, 
                                          se_ratio, 
                                          drop_rate, 
                                          activation=activation,
                                          signal_preserving=signal_preserving,
                                          normalization=normalization,
                                          alpha=alpha))

        self.blocks = nn.Sequential(*blocks)

        # head convolution
        self.head = nn.Sequential(
            nn.Conv2d(config[-1][1], feature_size, 1, 1, 0, bias=False),
            *([nn.BatchNorm2d(feature_size)] if normalization else [DynamicTanh(feature_size, False, alpha)]),
            activation(signal_preserving=signal_preserving)
            )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(variants[model_variant][3]),
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
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]  # number of input connections
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)  # Adjusted variance to 1/n
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]  # number of input connections
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)  # Adjusted variance to 1/n
                nn.init.zeros_(m.bias)
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.ones_(m.weight)
    #             nn.init.zeros_(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, mean=0.0, std=0.01)
    #             nn.init.zeros_(m.bias)

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
    model = DyTSNEfficientNet(model_variant="b0", 
                         num_classes=100, 
                         stem_channels=32, 
                         feature_size=1280, 
                         drop_connect_rate=0.2, 
                         activation=activation_class,
                         signal_preserving=False,
                         normalization=True,
                         alpha=1)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda")
    target = torch.randint(1, 100, (10,)).to("cuda")
    x = torch.randn(10, 3, 32, 32).to("cuda")
    output = model(x)
    loss = criterion(output, target)
    print(loss)
    print(output.shape)
    for name, param in model.named_parameters():
        print(name, param.shape)


    # Visualize model
    # dot = make_dot(model(x), params=dict(model.named_parameters()))
    # dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())