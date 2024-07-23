import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

class SignalLogger:
    """ SignalLogger class to log activations, weights and gradients for all layers in a PyTorch model """
    def __init__(self, model, layer_names=None):
        """
        Args:
            model (torch.nn.Module): PyTorch model
            layer_names (list): List of layer names for which activations, weights and gradients should be logged

        Example:
            model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
            signal_logger = SignalLogger(model)
            hooks = signal_logger.register_hooks()

            x = torch.randn(1, 3, 224, 224)
            model(x) # the hooks will be called during the forward pass

            signal_logger.extract_weights()
            signal_logger.update_running_stats()
            stats = signal_logger.get_signal_statistics()
            signal_logger.clear()
            for hook in hooks:
                hook.remove()
            print(stats, len(stats))
        """
        self.model = model
        self.layer_names = layer_names or [name for name, module in model.named_modules() if len(list(module.children())) == 0]
        self.batch_activations = {name: [] for name in self.layer_names}
        self.batch_weights = {name: [] for name in self.layer_names}
        self.batch_gradients = {name: [] for name in self.layer_names}
        self.running_stats = {name: {'mean_act': 0.0, 'std_act': 0.0,
                                     'mean_weight': 0.0, 'std_weight': 0.0,
                                     'mean_grad': 0.0, 'std_grad': 0.0}
                              for name in self.layer_names}
        self.batch_counter = {name: 0 for name in self.layer_names}

    def _hook_fn(self, module, input, output):
        module_name = self._get_module_name(module)
        if module_name in self.layer_names:
            self.batch_activations[module_name].append(output.detach().cpu().numpy())

    def _full_backward_hook_fn(self, module, grad_input, grad_output):
        module_name = self._get_module_name(module)
        if module_name in self.layer_names:
            self.batch_gradients[module_name].append(grad_output[0].detach().cpu().numpy())

    def _get_module_name(self, module):
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return None

    def register_hooks(self):
        """ Register forward and full backward hooks for all layers in the model, executed during forward and backward pass """
        hooks = []
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hooks.append(module.register_forward_hook(self._hook_fn))
                hooks.append(module.register_full_backward_hook(self._full_backward_hook_fn))
        return hooks

    def extract_weights(self):
        """ Extract weights for all layers in the model """
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                params = list(module.parameters())
                if params:
                    parameters = parameters_to_vector(params)
                    self.batch_weights[name].append(parameters.detach().cpu().numpy())

    def _update_stats(self, batch_data, layer, stat_prefix):
        if batch_data:
            concatenated_data = np.concatenate(batch_data, axis=0)
            mean = np.mean(concatenated_data)
            std = np.std(concatenated_data)
            self.running_stats[layer][f'mean_{stat_prefix}'] += mean
            self.running_stats[layer][f'std_{stat_prefix}'] += std
            batch_data.clear()

    def update_running_stats(self):
        """ Update running statistics for all layers 
        This function should be called after extracting weights and activations for all layers in the model 
       """
        for layer in self.layer_names:
            self._update_stats(self.batch_activations[layer], layer, 'act')
            self._update_stats(self.batch_weights[layer], layer, 'weight')
            self._update_stats(self.batch_gradients[layer], layer, 'grad')
            self.batch_counter[layer] += 1

    def get_signal_statistics(self):
        """ Get the mean and standard deviation of activations, weights and gradients for all layers 
        Returns:
            dict: Dictionary containing mean and standard deviation of activations, weights and gradients for all layers
            structured as follows: {layer_name: 
                                                {'mean_act': float, 
                                                'std_act': float, 
                                                'mean_weight': float, 
                                                'std_weight': float, 
                                                'mean_grad': float, 
                                                'std_grad': float}}
        """
        stats = {}
        for layer in self.layer_names:
            stats[layer] = {
                'mean_act':     self.running_stats[layer]['mean_act'] / self.batch_counter[layer],
                'std_act':      self.running_stats[layer]['std_act'] / self.batch_counter[layer],
                'mean_weight':  self.running_stats[layer]['mean_weight'] / self.batch_counter[layer],
                'std_weight':   self.running_stats[layer]['std_weight'] / self.batch_counter[layer],
                'mean_grad':    self.running_stats[layer]['mean_grad'] / self.batch_counter[layer],
                'std_grad':     self.running_stats[layer]['std_grad'] / self.batch_counter[layer],
            }
        return stats

    def clear(self):
        """ Clear all data stored in the logger """
        self.batch_activations = {name: [] for name in self.layer_names}
        self.batch_weights = {name: [] for name in self.layer_names}
        self.batch_gradients = {name: [] for name in self.layer_names}
        self.batch_counter = {name: 0 for name in self.layer_names}
        self.running_stats = {name: {'mean_act': 0.0, 'std_act': 0.0,
                                     'mean_weight': 0.0, 'std_weight': 0.0,
                                     'mean_grad': 0.0, 'std_grad': 0.0}
                              for name in self.layer_names}

# Example usage
if __name__ == "__main__":
    import torchvision
    model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")  # Updated for latest torchvision API
    signal_logger = SignalLogger(model)
    hooks = signal_logger.register_hooks()
    x = torch.randn(1, 3, 224, 224)
    model(x) # the hooks will be called during the forward pass
    signal_logger.extract_weights()
    signal_logger.update_running_stats()
    stats = signal_logger.get_signal_statistics()
    
    signal_logger.clear()
    for hook in hooks:
        hook.remove()
    
    print(stats, len(stats))
    #  now plot the mean activations and weights with respect to the layers
    import matplotlib.pyplot as plt
    layer_names = list(stats.keys())
    mean_acts = [stats[layer]['mean_act'] for layer in layer_names]
    mean_weights = [stats[layer]['mean_weight'] for layer in layer_names]
    mean_grads = [stats[layer]['mean_grad'] for layer in layer_names]
    plt.plot(mean_acts, label='Mean Activations')
    plt.plot(mean_weights, label='Mean Weights')
    plt.plot(mean_grads, label='Mean Gradients')
    plt.legend()
    plt.show()
