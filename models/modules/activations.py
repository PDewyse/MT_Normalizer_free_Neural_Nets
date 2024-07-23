import importlib
import torch
import torch.nn as nn

def load_activation_class(module_name, class_name):
    """Dynamically load an activation class from a module."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot find activation class '{class_name}' in module '{module_name}': {e}")

def compute_sp_const(activation_func, samples=(10240, 2560)):
    """
    Compute the variance-preserving constant for a given activation function using PyTorch.
    
    Args:
        activation_func: The activation function to apply.
        samples: The shape of the random noise sample.
    
    Returns:
        gamma: The variance-preserving constant.
    """
    x = torch.randn(samples)
    # mean_before, std_before = torch.mean(x), torch.std(x)

    y = activation_func(x)
    # mean_after, std_after = torch.mean(y), torch.std(y)

    # print(f"Mean before: {mean_before:.4f}, std before: {std_before:.4f}")
    # print(f"Mean after: {mean_after:.4f}, std after: {std_after:.4f}")
    gamma = torch.mean(torch.var(y, dim=1)).pow(-0.5)
    return gamma.item()

class Swish(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(Swish, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(lambda x: x * torch.sigmoid(x))
            # with numerical integration instead of sampling
            self.sp_const = 1/0.596469

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x) * self.sp_const

class HSwish(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(HSwish, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(lambda x: nn.ReLU6()(x + 3) * (1.0 / 6.0) * x)

        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * self.relu6(x + 3.0) * (1.0 / 6.0) * self.sp_const

class Sigmoid(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(Sigmoid, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(torch.sigmoid)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x) * self.sp_const

class HSigmoid(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(HSigmoid, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(lambda x: nn.ReLU6()(x + 3) * (1.0 / 6.0))

        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.0) * (1.0 / 6.0) * self.sp_const

class ReLU(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(ReLU, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(nn.ReLU())

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x) * self.sp_const

class SELU(nn.Module):
    def __init__(self, signal_preserving: bool = False):
        super(SELU, self).__init__()

        self.sp_const = 1
        if signal_preserving:
            self.sp_const = compute_sp_const(nn.SELU()) # NOTE: SELU is inherently variance preserving

        self.selu = nn.SELU()

    def forward(self, x):
        return self.selu(x) * self.sp_const

# example usage
if __name__ == "__main__":
    activations = ["Swish", "HSwish", "Sigmoid", "HSigmoid", "ReLU", "SELU"]
    module_name = "activations"

    for activation_name in activations:
        activation_class = load_activation_class(module_name, activation_name)
        activation = activation_class()
        print(f"{activation_name} VP const:", activation.sp_const)
        print(activation)