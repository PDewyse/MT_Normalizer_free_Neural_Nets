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

class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1)) 
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1,2,3], keepdims=True)
        var = torch.var(self.weight, axis=[1,2,3], keepdims=True)
        scale = 1 / torch.sqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain
    
    def forward(self, x):
        return nn.functional.conv2d(x, 
                                    weight=self.standardized_weights(), 
                                    bias=self.bias, 
                                    stride=self.stride, 
                                    padding=self.padding,
                                    dilation=self.dilation, 
                                    groups=self.groups)

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio, activation=Swish, signal_preserving=False):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')
        squeeze_channels = int(squeeze_channels)

        self.se_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_reduce = WSConv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.se_act = activation(signal_preserving=signal_preserving)
        self.se_expand = WSConv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.se_act_scale = nn.Sigmoid()

    def forward(self, x):
        out = self.se_avgpool(x)
        out = self.se_act(self.se_reduce(out))
        scale = self.se_act_scale(self.se_expand(out)) # Place attention on every channel
        return x * scale * 2 # factor 2 based on A. Brock paper

class Block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 expand_ratio, 
                 se_ratio, 
                 drop_connect_rate,
                 alpha = 0.2,
                 beta = 1.0,
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(Block, self).__init__()

        expand_channels = in_channels * expand_ratio
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        self.norm = normalization
        self.activation = activation(signal_preserving=signal_preserving)
        self.alpha = alpha
        self.beta = beta

        conv = []

        # Point wise convolution phase
        if expand_ratio != 1:
            pointwise_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                *([nn.BatchNorm2d(expand_channels)] if self.norm else []),
                activation(signal_preserving=signal_preserving)
                )
            conv.append(pointwise_conv1)

        # Depth wise convolution phase
        depthwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size, stride, kernel_size // 2, groups=expand_channels, bias=False),
            *([nn.BatchNorm2d(expand_channels)] if self.norm else []),
            activation(signal_preserving=signal_preserving)
            )
        conv.append(depthwise_conv)

        # Squeeze and excite phase
        if se_ratio != 0:
            conv.append(SqueezeAndExcite(expand_channels, in_channels, se_ratio, activation=activation, signal_preserving=signal_preserving))

        # Projection phase
        pointwise_conv2 = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            *([nn.BatchNorm2d(out_channels)] if self.norm else [])
            )
        conv.append(pointwise_conv2)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.activation(x) / self.beta
        
        if self.residual_connection:
            main_path = _drop_path(self.conv(out), self.drop_connect_rate, self.training)
            return x + self.alpha * main_path
        else:
            return self.alpha * self.conv(out)

class UNEfficientNet(nn.Module):
    def __init__(self, 
                 model_variant="b0", 
                 num_classes=100, 
                 stem_channels=32, 
                 feature_size=1280, 
                 drop_connect_rate=1, 
                 alpha=0.2,
                 activation=Swish, # Pass the class, not the instance
                 signal_preserving=False,
                 normalization=True):
        super(UNEfficientNet, self).__init__()
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
            *([nn.BatchNorm2d(stem_channels)] if normalization else []),
            activation(signal_preserving=signal_preserving)
            )

        # pre-compute the expected empirical variance for every residual block (A. Brock)
        # var_0 = 1
        # var_1 = var_0 + alpha ** 2
        # var_2 = var_1 + alpha ** 2 --> it just keeps adding alpha ** 2 PER block
        # TODO: alpha = nn.Parameter(torch.tensor(0.2, requires_grad=True))
        total_blocks = sum(conf[6] for conf in config)
        betas = [(1.0 + alpha ** 2 * i) ** 0.5 for i in range(total_blocks)]
        # betas = []
        # expected_std = 1.0
        # for _ in range(total_blocks):
        #     betas.append(1.0/expected_std)
        #     expected_std = (expected_std **2 + alpha**2)**0.5

        # construct the blocks based on the configuration of the specific model variant
        b_counter = 0
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
                                    alpha=alpha,
                                    beta=betas[b_counter],
                                    activation=activation, 
                                    signal_preserving=signal_preserving, 
                                    normalization=normalization))
            b_counter += 1
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(Block(out_channels,
                                        out_channels,
                                        kernel_size,
                                        1,
                                        expand_ratio,
                                        se_ratio,
                                        drop_rate,
                                        alpha=alpha,
                                        beta=betas[b_counter],
                                        activation=activation,
                                        signal_preserving=signal_preserving,
                                        normalization=normalization))
                b_counter += 1

        self.blocks = nn.Sequential(*blocks)

        # head convolution
        self.head = nn.Sequential(
            nn.Conv2d(config[-1][1], feature_size, 1, 1, 0, bias=False),
            *([nn.BatchNorm2d(feature_size)] if normalization else []),
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n)) # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

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
    model = UNEfficientNet(model_variant="b0", 
                           num_classes=100, 
                           stem_channels=32, 
                           feature_size=1280, 
                           drop_connect_rate=0.2,
                           activation=activation_class,
                           signal_preserving=True,
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
    # layer_names = [name for name, module in model.named_modules() if len(list(module.children())) == 0]
    # print("\n".join(layer_names))

    # dot = make_dot(model(x), params=dict(model.named_parameters()))
    # dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())