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
        Important: 
            The model should be created with inplace=False for activations, otherwise the hook will not work.
        Example usage:
        ```python
            import torch
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(224, 100),
                nn.ReLU(inplace=False), # inplace=False is important for hook to work
                nn.Linear(100, 10)
            )
            signal_logger = SignalLogger(model)
            hooks = signal_logger.register_hooks()

            for _ in range(10):
                inputs = torch.randn(32, 224)
                outputs = model(inputs)
                loss = outputs.sum()
                loss.backward()
                signal_logger.extract_weights()
                signal_logger.update_running_stats()

            for hook in hooks:
                hook.remove()

            signal_stats = signal_logger.get_signal_statistics()
            print(signal_stats)
            signal_logger.clear()
        ```
        """
        self.model = model
        # Get all layers without children (i.e. leaf nodes)
        # self.layer_names = layer_names or [f"{name} {module.__class__.__name__}" for name, module in model.named_modules() if len(list(module.children())) == 0]

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
                # params = list(module.parameters()) # TODO: Use all params or specific ones?
                # options: weight, bias, (gain), (alpha), ... (for bn weight:=gamma, bias:=beta)
                # specific learnable params can be filtered automatically by naming them accordingly (e.g.: residual_weight)
                # be careful because it looks for the substring weight!
                params = [p for n, p in module.named_parameters() if 'weight' in n]
                if params:
                    parameters = parameters_to_vector(params)
                    self.batch_weights[name].append(parameters.detach().cpu().numpy())

    def _update_stats(self, batch_data, layer, stat_prefix):
        if batch_data:
            concatenated_data = np.concatenate(batch_data, axis=0)
            if concatenated_data.ndim <= 2: # 1 square batch values, 2 mean over all
                mean = np.mean(concatenated_data**2)
                var = np.var(concatenated_data)
            elif concatenated_data.ndim == 4: # 1 mean over batch and spatial, 2 square them, 3 mean over channels
                mean = np.mean(np.mean(concatenated_data, axis=(0,2,3))**2) 
                var = np.mean(np.var(concatenated_data, axis=(0,2,3)))
            else:
                raise ValueError(f"Unsupported shape: {concatenated_data.shape}")
            
            self.running_stats[layer][f'mean_{stat_prefix}'] += mean
            self.running_stats[layer][f'std_{stat_prefix}'] += var
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
    import torch
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(224, 100),
        nn.ReLU(inplace=False), # inplace=False is important for hook to work
        nn.Linear(100, 10)
    )
    signal_logger = SignalLogger(model)
    hooks = signal_logger.register_hooks()

    # give an example training loop
    for _ in range(10):
        inputs = torch.randn(32, 224)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        signal_logger.extract_weights()
        signal_logger.update_running_stats()

    for hook in hooks:
        hook.remove()

    signal_stats = signal_logger.get_signal_statistics()
    print(signal_stats)
    signal_logger.clear()


    
    
