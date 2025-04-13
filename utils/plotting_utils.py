import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def beautify_layer_name(layer_name):
    """very specific function to make the layer names more presentable, will vary for each model"""
    # for cnn (no bn)
    last_part = layer_name.split(".")[-1]
    snd_last_part = layer_name.split(".")[-2]
    try:
        int(last_part)
        if last_part in "036" and snd_last_part == "feature_extractor":
            return "Conv"
        if last_part in "258" and snd_last_part == "feature_extractor":
            return "MaxPool"
        # if last_part in "159" and snd_last_part == "feature_extractor":
        #     return "bn"
        if last_part == "9":
            return "Flatten"
        if last_part in "036" and snd_last_part == "classifier":
            return "Linear"
        if last_part in "14" and snd_last_part == "classifier":
            return "bn"

    except ValueError:
        return last_part
    # for fnnns
    # last_part = layer_name.split(".")[-1]
    # try:
    #     int(last_part)
    #     if last_part == "0":
    #         return layer_name.split(".")[-2]
    #     if last_part == "1":
    #         return "bn"
    #     return layer_name.split(".")[-2]
    # except ValueError:
    #     return last_part
    
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

def plot_epoch_signal(df_mean, df_std, metrics, epochs, layers=None, save_path=None, im_description="", fig_size_modifier=30, debug=False, use_errorbars=False, **kwargs):
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
    debug : bool, optional
        plots for paper or debugging. The default is True. changes the x-axis labels, title, error bars, and size of the plot.
    **kwargs : dict
        Additional keyword arguments for the plot.
    """
    possible_metrics = ['mean_act', 'std_act', 
                        'mean_weight', 'std_weight',
                        'mean_grad', 'std_grad']
    for metric in metrics:
        if metric not in possible_metrics:
            raise ValueError(f"Metric: {metric} not found in the possible metrics: {possible_metrics}")
    
    if not all(epoch in df_mean['epoch'].values for epoch in epochs):
        raise ValueError("One or more epochs not found in the DataFrame epochs.")
    
    if layers is None:  # use all layers
        layers = df_mean.columns[1:]

    for layer in layers:
        if layer not in df_mean.columns:
            raise ValueError(f"Layer: {layer} not found in the DataFrame columns.")
        else:
            for epoch in epochs:
                layer_data = df_mean[layer][epoch]
                if not isinstance(layer_data, dict):
                    raise ValueError(f"Layer data for layer: {layer} on epoch: {epoch} must be a dictionary.")
    
    # Plot the metrics in subplots in a grid as square as possible
    num_plots = len(metrics)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Ensure 16:9 aspect ratio
    if not debug:
        fig_size_modifier = 7 # for FNNNs and efficientnets and cnns

    width = fig_size_modifier * num_cols
    height = width * 8 / 10 # 6/10 for FNNNs and 8/10 efficientnets # 7/10 for CNNs

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for epoch in epochs:
            layer_metric_mean = np.array([df_mean[layer][epoch][metric] for layer in layers])
            layer_metric_std = np.array([df_std[layer][epoch][metric] for layer in layers])

            if use_errorbars and f'std_{metric.split("_")[1]}' in metrics and metric.split("_")[0] == "mean":
                std_metric = [df_mean[layer][epoch][f'std_{metric.split("_")[1]}'] for layer in layers]
                ax.errorbar(np.arange(len(layers)), layer_metric_mean, yerr=std_metric, **kwargs)
                y_min = min(layer_metric_mean - np.array(std_metric))
                y_max = max(layer_metric_mean + np.array(std_metric))
                margin = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)
            if debug:
                ax.plot(layers, layer_metric_mean, label=f'Epoch {epoch}', **kwargs)
            if not debug:
                # ax.plot(layers, layer_metric_mean, label=f'Epoch {epoch}', **kwargs)
                ax.plot(np.arange(len(layers)), layer_metric_mean, label=f'Epoch {epoch}', **kwargs)
                # ax.errorbar(layers, layer_metric_mean, yerr=layer_metric_std, fmt='-o', capsize=5, label='_nolegend_')
                # ax.fill_between(layers, 
                #     layer_metric_mean - layer_metric_std, 
                #     layer_metric_mean + layer_metric_std, 
                #     color='b', alpha=0.2, label='_nolegend_')
            
            # Add horizontal line at 0 for mean_act and mean_weight, and at 1 for std_act and std_weight
            ax.axhline(0, color='black', linestyle='dotted')
            # if metric in ['mean_act', 'mean_weight', 'mean_grad', 'std_grad']:
            #     ax.axhline(0, color='black', linestyle='dotted')
            # elif metric in ['std_act', 'std_weight']:
            #     ax.axhline(1, color='black', linestyle='dotted')
            #     ax.axhline(0, color='red', linestyle='dotted')
        
        if debug:
            ax.set_title(f"{metric}")
            ax.set_xticks(np.arange(len(layers)))
            ax.set_xticklabels(layers, rotation=90, ha='right')#, fontsize=12)
            ax.set_xlabel("Layer", fontsize=14)
            # ax.set_ylim(0, 1e-8)
        else:
            # with compacting the layer names for FFNN
            # label_size = 10
            # general_size = 13
            # ax.set_xticks(np.arange(len(layers)))
            # # make the layer names less technical
            # compact_layers = [beautify_layer_name(layer) for layer in layers]
            # ax.set_xticklabels(compact_layers, rotation=45, ha='right', fontsize=label_size-1)
            # ax.tick_params(axis='y', labelsize=label_size)

            # with compacting the layer names for EfficientNet
            label_size = 11
            general_size = 15
            ax.set_xticks(np.arange(0, len(layers), 50))
            ax.set_xticklabels(np.arange(0, len(layers), 50), fontsize=label_size)
            ax.tick_params(axis='y', labelsize=label_size)
            # without compacting the layer names
            ax.set_xticks(np.arange(0, len(layers), 20))
            ax.set_xticklabels(np.arange(0, len(layers), 20))
            
            # if "act" in metric:
            #     ax.set_ylim(0, 1)
            ax.set_xlabel("Model Depth (layers)", fontsize=general_size)
            # ax.set_ylim(0, 200)

        ylabel = "ACSM" if "mean" in metric else "ACV"
        metric_name = "Activations" if "act" in metric else "Weights" if "weight" in metric else "Gradients"
        ax.set_ylabel(f"{ylabel} of {metric_name}")#, fontsize=general_size-1)
        ax.legend()#fontsize=general_size-1)
        
    # Remove empty subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if save_path:
        plot_path = os.path.join(save_path, f"{"debug_" if use_debugging else ""}epoch_signals{'_' + im_description}.png")
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
        for i, layer in enumerate(layers):
            epoch_metric = [df[layer][epoch][metric] for epoch in num_epochs]
            # print(layer,epoch_metric)
            # addition = f"block {i} " if "linear" in layer else ""
            # ax.plot(epoch_numbers, epoch_metric, label=f'{addition}{beautify_layer_name(layer)}', **kwargs)
            ax.plot(epoch_numbers, epoch_metric, label=layer, **kwargs)
        
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
def parse_log_file(log_path, verbose=False) -> pd.DataFrame:
    import ast
    with open(log_path, 'r') as f:
        lines = f.readlines()

    general_info = lines[0].strip()
    if verbose: 
        print(f"Parsing log file: {general_info}")

    columns = lines[1].strip().split('\t')
    if verbose:
        print("Columns:" + "\n".join(columns))

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
    if verbose:
        print(df.info())
        print(df.head())
    return df

def calculate_mean_std(dfs, metrics, epochs=None, layers=None):
    # Calculate the mean and standard deviation of the dataframes containing tuples of metrics per layer
    
    possible_metrics = ['mean_act', 'std_act', 
                        'mean_weight', 'std_weight',
                        'mean_grad', 'std_grad']
    for metric in metrics:
        if metric not in possible_metrics:
            raise ValueError(f"Metric: {metric} not found in the possible metrics: {possible_metrics}")
    
    if epochs is None:  # use all epochs
        epochs = dfs[0]['epoch'].values

    for df in dfs:
        if not all(epoch in df['epoch'].values for epoch in epochs):
            raise ValueError("One or more epochs not found in the DataFrame epochs.")
    
        if layers is None:  # use all layers except the first column with the epochs
            layers = df.columns[1:]

        for layer in layers:
            if layer not in df.columns:
                raise ValueError(f"Layer: {layer} not found in the DataFrame columns.")
            else:
                for epoch in epochs:
                    layer_data = df[layer][epoch]
                    if not isinstance(layer_data, dict):
                        raise ValueError(f"Layer data for layer: {layer} on epoch: {epoch} must be a dictionary.")
    
    train_layers_mean_df = dfs[0].copy()
    train_layers_std_df = dfs[0].copy()

    # Overwrite values with computed mean and std
    for metric in metrics:
        for epoch in epochs:
            for layer in layers:
                values = [df[layer][epoch][metric] for df in dfs]  # Extract values from all data frames
                train_layers_mean_df[layer][epoch][metric] = np.mean(values)
                train_layers_std_df[layer][epoch][metric] = np.std(values)

    return train_layers_mean_df, train_layers_std_df

if __name__ == "__main__":
    # Define the paths based on the provided variables
    script_dir = os.path.dirname(os.path.abspath(__file__)) # root/utils
    root_dir = os.path.dirname(script_dir) # root

    model_name = "WSSNEfficientNet1"
    use_debugging = False
    use_errorbars = False
    # get all folder names under the root folder
    folder_dir = os.path.join(root_dir, "checkpoints", model_name)
    run_folders = os.listdir(folder_dir)
    run_folders = ["2025-04-05_13-06-42_spp run"]
    # 2024-12-08_05-46-13_test 5 fold spp bn and lr
      
    for i in range(len(run_folders)):
        print(f"Processing run: {run_folders[i]} ({i+1}/{len(run_folders)})")
        # Parse the log files
        dfs_layers = []
        for j in range(len(run_folders)):
            train_run_dir = os.path.join(root_dir, "checkpoints", model_name, run_folders[j])
            train_df = parse_log_file(os.path.join(train_run_dir, "training_log.txt"), verbose=False)
        
            # split the data frame into two data frames, one for the metrics such as accuracy etc and the other for the layer activations and weights
            train_metrics = ['train_loss','val_loss', 'accuracy','balanced_accuracy','precision','recall','f1','roc_auc']
            train_metrics_df = train_df[train_metrics]
            train_metrics_df['epoch'] = train_df['epoch'] # keep the epochs

            train_layers_df = train_df.drop(columns=train_metrics)
            layer_metrics = ['mean_act', 'std_act', 'mean_weight', 'std_weight', 'mean_grad', 'std_grad']
            dfs_layers.append(train_layers_df)
        
        # Calculate the mean and standard deviation of the data frames
        # train_layers_mean_df, train_layers_std_df = calculate_mean_std(dfs_layers, layer_metrics)

        # Plot the training and testing metrics
        save_path = os.path.join(train_run_dir, "figures")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_description = run_folders[i].split("_")[-1]
        plot_metrics(df=train_metrics_df,
                        metrics=train_metrics, 
                        fig_size_modifier=10, 
                        save_path=save_path, 
                        im_description=image_description)
        plot_epoch_signal(df_mean=train_layers_df,
                            df_std=train_layers_df,
                            metrics=layer_metrics, 
                            epochs=[0,9,19],
                            fig_size_modifier=50, 
                            save_path=save_path, 
                            im_description=image_description,
                            debug=use_debugging,
                            use_errorbars=use_errorbars)
        break
        # we don't need this stuff for paper,
        # and for error bars between runs we use the runs as repeats not to plot separately
        # it's janky but it works
        plot_layer_signal(df=train_layers_df, 
                        metrics=layer_metrics, 
                        layers=['feature_extractor.0', 'feature_extractor.4','feature_extractor.5','feature_extractor.11','classifier.0','classifier.2','classifier.4'],
                        fig_size_modifier=10, 
                        save_path=save_path,
                        im_description=image_description)
        # for spp test FFNN #["head.0","blocks.0.linear.0","blocks.4.linear.0","blocks.7.linear.0","tail.0"],#['blocks.7.conv.0.0','blocks.7.conv.1.0','blocks.7.conv.2.se_reduce','blocks.7.conv.2.se_expand','blocks.7.conv.3.0'],#["blocks.0.linear.0", "blocks.0.linear.1", "blocks.4.linear.1","blocks.4.linear.2.relu", "blocks.5.linear.0", "blocks.5.linear.1", "blocks.9.linear.0"],
        # for spp test cnn # ['feature_extractor.0', 'feature_extractor.3','feature_extractor.6','classifier.0','classifier.2','classifier.4']
        # for spp test CNN1 #
        # for spp test CNN2 #
        # example plots
        # plot_metrics(train_df, ["train_loss", "accuracy"], fig_size_modifier=3, save_path=save_path)
        # plot_epoch_signal(train_layers_df, layer_metrics, [0,10,19], layers=None, fig_size_modifier=15, save_path=save_path)
        # plot_layer_signal(train_layers_df, layer_metrics, ["blocks.0.conv.1.se_reduce", "blocks.10.conv.2.se_reduce", "blocks.15.conv.2.se_reduce"], num_epochs=None, fig_size_modifier=4, save_path=save_path)
      # if repeat_test:
    #     repeat_dfs = []
    #     for i in range(len(run_folders)):
    #         # Parse the log files
    #         train_run_dir = os.path.join(root_dir, "checkpoints", model_name, run_folders[i])
    #         train_df = parse_log_file(os.path.join(train_run_dir, "training_log.txt"), verbose=True)
    #         repeat_dfs.append(train_df)
        
    #     # Check if the data frames are the same
    #     for i in range(1, len(repeat_dfs)):
    #         if not repeat_dfs[i].equals(repeat_dfs[0]):
    #             raise ValueError(f"DataFrames {i} and 0 are not equal.")
        
    #     # calculate the mean and standard deviation of duplicates
    #     print(repeat_dfs[0].head())
    #     # combined_repeat_df = pd.concat(repeat_dfs)
    #     # mean_repeat_df = combined_repeat_df.mean()
    #     # std_repeat_df = combined_repeat_df.std()

    #     # print(repeat_dfs[0].head())
    #     # print(mean_repeat_df.head())
    #     # print(std_repeat_df.head())
    #     quit()      