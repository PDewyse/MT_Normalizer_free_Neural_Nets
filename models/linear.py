import torch
import torch.nn as nn

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
else:
    from models.modules.activations import Swish

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=Swish(), normalization=True):
        super(LinearBlock, self).__init__()
        self.normalization = normalization
        self.linear = nn.Linear(in_features, out_features)
        if self.normalization:
            self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation # NOTE: activation is a class

    def forward(self, x):
        x = self.linear(x)
        if self.normalization:
            x = self.bn(x)
        x = self.activation(x)
        return x

class Linear(nn.Module):
    """ A simple linear model with a configurable number of blocks for increasing depth. and normalization."""
    def __init__(self, out_features=100, hidden_features=512, num_blocks=100, activation=Swish(), normalization=True):
        super(Linear, self).__init__()

        self.normalization = normalization
        self.hidden_features = hidden_features
        self.out_features = out_features
        
        self.flatten = nn.Flatten() # because we use it on cifar100 images

        self.head = [nn.LazyLinear(hidden_features)]
        if self.normalization:
            self.head.append(nn.BatchNorm1d(hidden_features))
        self.head.append(activation)
        self.head = nn.Sequential(*self.head)

        hidden_blocks = []
        for _ in range(num_blocks):
            hidden_blocks.append(LinearBlock(hidden_features, 
                                             hidden_features, 
                                             activation=activation, 
                                             normalization=self.normalization))
        
        self.blocks = nn.Sequential(*hidden_blocks)

        self.tail = nn.Sequential(
            nn.Linear(hidden_features, out_features)
        )
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.head(out)
        out = self.blocks(out)
        out = self.tail(out)
        return out

# example usage
if __name__ == "__main__":
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    activation_class = load_activation_class('modules.activations', 'SELU')
    activation = activation_class(signal_preserving=False)
    model = Linear(activation=activation, normalization=False)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to("cuda")
    target = torch.randint(1, 100, (10,)).to("cuda")
    x = torch.randn(10, 3, 32, 32).to("cuda")
    output = model(x)
    loss = criterion(output, target)
    print(loss)
    print(output.shape)

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())

    
