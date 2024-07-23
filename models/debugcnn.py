import torch
import torch.nn as nn
from models.modules.activations import load_activation_class
    
class DebugCNN(nn.Module):
    def __init__(self, in_channels, out_features, hidden_features, activation, signal_preserving=False):
        super(DebugCNN, self).__init__()
        
        # Dynamically load the activation class
        activation_class = load_activation_class('models.modules.activations', activation)  # Replace 'your_module_name' with the actual module name
        self.activation = activation_class(signal_preserving=signal_preserving)

        self.conv1 = nn.Conv2d(in_channels, hidden_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.fc1 = nn.LazyLinear(out_features)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Example usage
if __name__ == '__main__':
    model = DebugCNN(in_channels=3, out_features=100, hidden_features=64, activation='Swish', signal_preserving=False)
    print(model)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(output.shape)