import os
from datetime import datetime
import argparse
import warnings
import wandb
import json
import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets

from utils.data_utils import DatasetCNN, set_seed, get_transforms, get_split
from utils.trainer import Trainer
from utils.tester import Tester

# Helper fuctions
def get_with_warning(config, key, default, config_name="config"):
    """Get a value from the config, with a warning if the default is used."""
    def custom_warning(message):
            return warnings.warn(message, stacklevel=3)
    
    if key not in config:
        warning_str = f"{config_name} missing key '{key}', using default value: {default}"
        custom_warning(warning_str)
        
    return config.get(key, default)

def load_class(module_name, class_name):
    """Dynamically load a model class from a module. make sure that the module correctly names the class.
    
    Args:
        module_name (str): Name of the module containing the class.
        class_name (str): Name of the class to load.
        
    Returns:
        class: The class object."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        error_message = f"Cannot find class '{class_name}' in module '{module_name}': {e}"
        raise ImportError(error_message)

def main(args):
    # Load hyperparameters from the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract configuration values
    seed = get_with_warning(config, 'seed', 34)
    model_name = get_with_warning(config, 'model', 'EfficientNet')
    description = get_with_warning(config, 'description', 'test')
    batch_size = get_with_warning(config, 'batch_size', 32)
    learning_rate = get_with_warning(config, 'learning_rate', 0.001)
    num_epochs = get_with_warning(config, 'num_epochs', 10)
    use_scheduler = get_with_warning(config, 'use_scheduler', False)
    scheduler_step_size = get_with_warning(config, 'scheduler_step_size', 10)
    scheduler_gamma = get_with_warning(config, 'scheduler_gamma', 0.1)
    save_every = get_with_warning(config, 'save_every', 1)
    log = get_with_warning(config, 'log', False) # True (WandB+local), False (on terminal)
    log_signal = get_with_warning(config, 'log_signal', False) # NOTE: huge performance hit!
    debug = get_with_warning(config, 'debug', True)
    # Tune parameters here
    normalization = get_with_warning(config, 'normalization', True)
    activation = get_with_warning(config, 'activation', 'Swish')
    signal_preserving = get_with_warning(config, 'signal_preserving', False)
    # alpha = get_with_warning(config, 'alpha', 0.5) # fDynamic tanh hyperparameter

    # Seed for torch and everything
    set_seed(seed)

    model_names = [
        'DebugCNN',
        'CNN', 
        'CNN1',
        'CNN2',
        'SNEfficientNet',
        'SNEfficientNet1',
        'UNEfficientNet',
        'EfficientNet', 
        'EfficientNet1',
        'EfficientNet1b',
        'EfficientNet2',
        'EfficientNet3',
        'EfficientNet3b',
        'EfficientNet4',
        'EfficientNet4b',
        'FFNN', 
        'NFEfficientNet', 
        'NFEfficientNetv2',
        'NFEfficientNetv3', 
        "NFEfficientNetv4",
        "SENet",
        "NFMobileNetv3",
        "DyTEfficientNet",
        "DyTSNEfficientNet",
        "WSEfficientNet",
        "WSSNEfficientNet",
        "WSSNEfficientNet1",
        "DyMPSNEfficientNet1"
        ] # TODO: find a way to automate this lookup via ast.parse
    
    if model_name not in model_names:
        raise ValueError(f"Model name '{model_name}' not found in available models: {model_names}")
    if '_' in description:
        raise ValueError("Description should not contain underscores, as it is used in plotting utils.")
    
    # Directory for saving checkpoints of specific models per run
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(script_dir, "checkpoints", model_name, f"{timestamp}_{description}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Training {model_name} ({description}) in directory: {save_dir}")
    print(f"Training configuration:\n{config}")

    if log:
        wandb.init(project=f"Training {model_name}", name=description, notes=description)
        wandb.config.update(config)
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config, f)
       
    if debug:
        # Make Dummy data
        train_data = DatasetCNN(torch.randn(1500, 3, 32, 32))
        val_data = DatasetCNN(torch.randn(1500, 3, 32, 32))
        test_data = DatasetCNN(torch.randn(1500, 3, 32, 32))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Load model class dynamically
        model_class = load_class(f'models.{model_name.lower()}', model_name)
        activation_class = load_class('models.modules.activations', activation)
        activation = activation_class(signal_preserving=signal_preserving)
        model = model_class(activation=activation, normalization=normalization)

    else:
        # Data preparation
        transform_train, transform_test = get_transforms()
        train_data = torchvision.datasets.CIFAR100(root=os.path.join(script_dir, "data"), train=True, download=True, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(root=os.path.join(script_dir, "data"), train=False, download=True, transform=transform_test)

        train_data, val_data = get_split(train_data, ratio=0.8)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Load model and activation classes dynamically
        print(f"Loading model: {model_name}")
        model_class = load_class(f'models.{model_name.lower()}', model_name)
        activation_class = load_class('models.modules.activations', activation) # Instantiate inside the model for plotting
        model = model_class(activation=activation_class, 
                            signal_preserving=signal_preserving, 
                            normalization=normalization)
        # model = torchvision.models.efficientnet_b0() # weights="IMAGENET1K_V1"
        # model.classifier = torch.nn.LazyLinear(100)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)# , weight_decay=1e-5)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Training
    trainer = Trainer(model, 
                      criterion, 
                      optimizer, 
                      train_loader, 
                      save_dir, 
                      scheduler=scheduler, 
                      save_every=save_every, 
                      log=log, 
                      log_signal=log_signal)
    trainer.train_model(num_epochs=num_epochs, val_loader=val_loader)
    # save the whole model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_args': {"activation": activation, "normalization": normalization, "signal_preserving": signal_preserving},
    }, os.path.join(save_dir, "final_model_everything.pt"))
    # Test the model
    tester = Tester(model, criterion, test_loader, save_dir, log=log)
    _ = tester.test()

    if log:
        print("Training configuration logged with WandB:")
        print(wandb.config)
        wandb.finish()
    else:
        print(f"Training configuration:\n{config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, default="config_training.json", help='Configuration file name')
    args = parser.parse_args()
    main(args)