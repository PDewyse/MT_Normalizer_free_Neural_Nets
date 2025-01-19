import math
import torch
import torch.nn as nn

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
else:
    from models.modules.activations import Swish

class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(LinearBlock, self).__init__()
        self.normalization = normalization
        
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            *([nn.BatchNorm1d(out_features, affine=False)] if self.normalization else []),
            activation(signal_preserving=signal_preserving)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class FFNN(nn.Module):
    """ A simple linear model with a configurable number of blocks for increasing depth. and normalization."""
    def __init__(self,
                 out_features=100,
                 hidden_features=100,
                 num_blocks=10,
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(FFNN, self).__init__()

        self.normalization = normalization
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        self.flatten = nn.Flatten() # because we use it on cifar100 images

        self.head = nn.Sequential(
            nn.Linear(3*32*32, hidden_features),
            *([nn.BatchNorm1d(hidden_features)] if self.normalization else []),
            activation(signal_preserving=signal_preserving)
        )
        
        hidden_blocks = []
        for _ in range(num_blocks-2): # -2 for head and tail
            hidden_blocks.append(LinearBlock(hidden_features,
                                            hidden_features,
                                            activation=activation,
                                            signal_preserving=signal_preserving,
                                            normalization=self.normalization))
    
        self.blocks = nn.Sequential(*hidden_blocks)

        self.tail = nn.Sequential(
            nn.Linear(hidden_features, out_features)
        )

        self._initialize_weights()
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.head(out)
        out = self.blocks(out)
        out = self.tail(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # m.weight.requires_grad = False  # Freeze gamma (scale)
                # m.bias.requires_grad = False    # Freeze beta (bias)

# example usage
if __name__ == "__main__":
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from torchviz import make_dot

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    activation_class = load_activation_class('modules.activations', 'SELU')
    model = FFNN(activation=activation_class, signal_preserving=False, normalization=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda")
    target = torch.randint(1, 100, (10,)).to("cuda")
    x = torch.randn(10, 3, 32, 32).to("cuda")
    output = model(x)
    loss = criterion(output, target)
    print(loss)
    print(output.shape)

    # Visualize model
    print("\nModel depth:")
    print(sum(1 for _ in model.modules()))

    # Visualize model
    print(model)

    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())

    
