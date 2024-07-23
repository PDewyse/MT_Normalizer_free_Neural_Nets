import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(df, metrics, save_path=None, im_description="", fig_size_modifier=10, **kwargs):
    """
    Plot the metrics in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the metrics.
    metrics : list
        The metrics to plot. Possible metrics are: ['train_loss', 'val_loss', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    save_path : str, optional
        The path to save the plot to. The default is None.
    im_description : str, optional
        The description of the image that will be added to the image file name. The default is "".
    fig_size_modifier : float, optional
        The figure size modifier. The default is 5.
    **kwargs : dict
        Additional keyword arguments for the plot.
    """
    # Check if the metrics are present in the DataFrame
    for metric in metrics:
        if metric not in df.columns:
            print(df.columns)
            raise ValueError(f"Metric: {metric} not found in the DataFrame columns.")
    
    # Plot the metrics in subplots in a grid as square as possible
    num_plots = len(metrics)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Ensure 16:9 aspect ratio
    width = fig_size_modifier * num_cols
    height = width * 9 / 16

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(df["epoch"], df[metric], **kwargs)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")

    # Remove empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        plot_path = os.path.join(save_path, f"metrics{'_' + im_description}.png")
        plt.savefig(plot_path)
        print(f"Saved plot to: {plot_path}")
    # plt.show()

def plot_epoch_signal(df, metrics, epochs, layers=None, save_path=None, im_description="", fig_size_modifier=30, **kwargs):
    """
    Plot the metrics for the layers at specific epochs.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the layer data.
    metrics : list
        The metrics to plot. Possible metrics are: ['mean_act', 'std_act', 'mean_weight', 'std_weight', 'mean_grad', 'std_grad']
    epochs : list
        The epochs to plot the data for, different epochs will be plotted in different colors.
    layers : list, optional
        The layers to plot the data for. If None, all layers are plotted. The default is None.
    save_path : str, optional
        The path to save the plot to. The default is None.
    im_description : str, optional
        The description of the image that will be added to the image file name. The default is "".
    fig_size_modifier : float, optional
        The figure size modifier. The default is 5.
    **kwargs : dict
        Additional keyword arguments for the plot.
    """
    possible_metrics = ['mean_act', 'std_act', 
                        'mean_weight', 'std_weight',
                        'mean_grad', 'std_grad']
    for metric in metrics:
        if metric not in possible_metrics:
            raise ValueError(f"Metric: {metric} not found in the possible metrics: {possible_metrics}")
    
    if not all(epoch in df['epoch'].values for epoch in epochs):
        raise ValueError("One or more epochs not found in the DataFrame epochs.")
    
    if layers is None:  # use all layers
        layers = df.columns[1:]

    for layer in layers:
        if layer not in df.columns:
            raise ValueError(f"Layer: {layer} not found in the DataFrame columns.")
        else:
            for epoch in epochs:
                layer_data = df[layer][epoch]
                if not isinstance(layer_data, dict):
                    raise ValueError(f"Layer data for layer: {layer} on epoch: {epoch} must be a dictionary.")
    
    # Plot the metrics in subplots in a grid as square as possible
    num_plots = len(metrics)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Ensure 16:9 aspect ratio
    width = fig_size_modifier * num_cols
    height = width * 9 / 16

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for epoch in epochs:
            layer_metric = [df[layer][epoch][metric] for layer in layers]
            if f'std_{metric.split("_")[1]}' in metrics and metric.split("_")[0] == "mean":
                std_metric = [df[layer][epoch][f'std_{metric.split("_")[1]}'] for layer in layers]
                ax.errorbar(layers, layer_metric, yerr=std_metric, label=f'Epoch {epoch}', **kwargs)
            else:
                ax.plot(layers, layer_metric, label=f'Epoch {epoch}', **kwargs)
            
            # Add horizontal line at 0 for mean_act and mean_weight, and at 1 for std_act and std_weight
            if metric in ['mean_act', 'mean_weight', 'std_grad']:
                ax.axhline(0, color='black', linestyle='dotted')
            elif metric in ['std_act', 'std_weight']:
                ax.axhline(1, color='black', linestyle='dotted')
                ax.axhline(0, color='red', linestyle='dotted')
        
        ax.set_title(f"{metric}")
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers, rotation=90, ha='right')
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.legend()
        
    # Remove empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        plot_path = os.path.join(save_path, f"epoch_signals{'_' + im_description}.png")
        plt.savefig(plot_path)
        print(f"Saved plot to: {plot_path}")
    # plt.show()

def plot_layer_signal(df, metrics, layers, num_epochs=None, save_path=None, im_description="", fig_size_modifier=10, **kwargs):
    """
    Plot the metrics for the epochs at specific layers.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the layer data.
    metrics : list
        The metrics to plot. Possible metrics are: ['mean_act', 'std_act', 'mean_weight', 'std_weight', 'mean_grad', 'std_grad']
    layers : list
        The layers to plot the data for.
    num_epochs : list, optional
        The epochs to plot the data for. If None, all epochs are plotted. The default is None.
    save_path : str, optional
        The path to save the plot to. The default is None.
    im_description : str, optional
        The description of the image that will be added to the image file name. The default is "".
    fig_size_modifier : float, optional
        The figure size modifier. The default is 5.
    **kwargs : dict
        Additional keyword arguments for the plot.
    """    
    possible_metrics = ['mean_act', 'std_act', 
                        'mean_weight', 'std_weight',
                        'mean_grad', 'std_grad']
    for metric in metrics:
        if metric not in possible_metrics:
            raise ValueError(f"Metric: {metric} not found in the possible metrics: {possible_metrics}")
    
    if not all(layer in df.columns for layer in layers):
        raise ValueError("One or more layers not found in the DataFrame columns.")
    
    if num_epochs is None:  # use all epochs
        num_epochs = df['epoch'].values

    for epoch in num_epochs:
        if epoch not in df['epoch'].values:
            raise ValueError(f"Epoch: {epoch} not found in the DataFrame epochs.")
        else:
            for layer in layers:
                epochs_data = df[layer][epoch]
                if not isinstance(epochs_data, dict):
                    raise ValueError(f"Epoch data for layer: {layer} on epoch: {epoch} must be a dictionary.")
    
    # Plot the metrics in subplots in a grid as square as possible
    num_plots = len(metrics)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Ensure 16:9 aspect ratio
    width = fig_size_modifier * num_cols
    height = width * 9 / 16

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    axes = axes.flatten()

    epoch_numbers = np.arange(len(num_epochs))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for layer in layers:
            epoch_metric = [df[layer][epoch][metric] for epoch in num_epochs]
            ax.plot(epoch_numbers, epoch_metric, label=f'Layer {layer}', **kwargs)
        
        ax.set_title(f"{metric}")
        # ax.set_xticks(epoch_numbers)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        
    # Remove empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        plot_path = os.path.join(save_path, f"layer_signals{'_' + im_description}.png")
        plt.savefig(plot_path)
        print(f"Saved plot to: {plot_path}")
    # plt.show()

# Function to read the log file and parse the data
def parse_log_file(log_path) -> pd.DataFrame:
    import ast
    with open(log_path, 'r') as f:
        lines = f.readlines()

    general_info = lines[0].strip()
    print(f"Parsing log file: {general_info}")

    columns = lines[1].strip().split('\t')
    print(f"Columns: {columns}")

    # Extract data from the remaining rows
    data = []
    for line in lines[2:]:
        row = line.strip().split('\t')
        parsed_row = []
        for i, item in enumerate(row):
            try:
                # layer activations and weights are stored as strings of dictionaries
                parsed_item = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                # other values are just stored as strings of floats
                parsed_item = item if i == 0 else float(item)
            
            parsed_row.append(parsed_item)
        data.append(parsed_row)
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

def parse_epoch_stats(epoch_stats) -> pd.DataFrame:
    pass

if __name__ == "__main__":
    # Define the paths based on the provided variables
    script_dir = os.path.dirname(os.path.abspath(__file__)) # root/utils
    root_dir = os.path.dirname(script_dir) # root

    model_name = "spefficientnet"
    run_folder = "2024-07-20_17-46-45_test sp effnet bn"
    # model_name = "efficientnet"
    # run_folder = "2024-07-09_16-29-53_Plot example"
    

    # Parse the log files
    train_run_dir = os.path.join(root_dir, "checkpoints", model_name, run_folder)
    train_df = parse_log_file(os.path.join(train_run_dir, "training_log.txt"))

    # split the data frame into two data frames, one for the metrics such as accuracy etc and the other for the layer activations and weights
    train_metrics = ['train_loss','val_loss', 'accuracy','balanced_accuracy','precision','recall','f1','roc_auc']
    train_metrics_df = train_df[train_metrics]
    train_metrics_df['epoch'] = train_df['epoch']

    train_layers_df = train_df.drop(columns=train_metrics)
    layer_metrics = ['mean_act', 'std_act', 'mean_weight', 'std_weight', 'mean_grad', 'std_grad']

    print(train_metrics_df.info())
    print(train_layers_df.info())

    # Plot the training and testing metrics
    save_path = os.path.join(train_run_dir, "figures")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_description = "sp eff nvidia"
    plot_metrics(df=train_metrics_df, 
                 metrics=train_metrics, 
                 fig_size_modifier=10, 
                 save_path=save_path, 
                 im_description=image_description)
    plot_epoch_signal(df=train_layers_df, 
                      metrics=layer_metrics, 
                      epochs=[0,9,19],
                      fig_size_modifier=50, 
                      save_path=save_path, 
                      im_description=image_description)
    plot_layer_signal(df=train_layers_df, 
                      metrics=layer_metrics, 
                      layers=["blocks.0.conv.1.se_expand","blocks.9.conv.2.se_expand", "blocks.14.conv.2.se_expand"], 
                      fig_size_modifier=10, 
                      save_path=save_path,
                      im_description=image_description)
    # example plots
    # plot_metrics(train_df, ["train_loss", "accuracy"], fig_size_modifier=3, save_path=save_path)
    # plot_epoch_signal(train_layers_df, layer_metrics, [0,10,19], layers=None, fig_size_modifier=15, save_path=save_path)
    # plot_layer_signal(train_layers_df, layer_metrics, ["blocks.0.conv.1.se_reduce", "blocks.10.conv.2.se_reduce", "blocks.15.conv.2.se_reduce"], num_epochs=None, fig_size_modifier=4, save_path=save_path)
    