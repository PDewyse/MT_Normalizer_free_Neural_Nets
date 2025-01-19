import torch
import torch.nn as nn

if __name__ == '__main__':
    from modules.activations import Swish, load_activation_class
else:
    from models.modules.activations import Swish    
class CNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_features=100,
                 feature_size=512,
                 activation=Swish,
                 signal_preserving=False,
                 normalization=True):
        super(CNN, self).__init__()

        self.normalization = normalization
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            *([nn.BatchNorm2d(32)] if not self.normalization else []),
            activation(signal_preserving=signal_preserving),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            *([nn.BatchNorm2d(64)] if not self.normalization else []),
            activation(signal_preserving=signal_preserving),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            *([nn.BatchNorm2d(128)] if not self.normalization else []),
            activation(signal_preserving=signal_preserving),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, feature_size),
            activation(signal_preserving=signal_preserving),
            nn.Linear(feature_size, int(feature_size/2)),
            activation(signal_preserving=signal_preserving),
            nn.Linear(int(feature_size/2), out_features)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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
    model = CNN(activation=activation_class, signal_preserving=True, normalization=True)
    
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
    for name in model.named_modules():
        print(name)

    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render(filename=model.__class__.__name__, directory="models/visualizations", format='png')

    # # Calculate number of parameters
    # print("\nNumber of parameters:")
    # summary(model, input_size=(10, 3, 32, 32))

    # # Calculate FLOPs
    # flops = FlopCountAnalysis(model, x)
    # print("\nFLOPs:")
    # print(flops.total())

