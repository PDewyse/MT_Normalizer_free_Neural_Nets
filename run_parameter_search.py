import os
import subprocess
import json
import itertools
import gc
import argparse
def print_hyperparameters(config):
    print("==============================================")
    print("Fixed hyperparameters:")
    for key, value in config["fixed"].items():
        print(f"{key}: {value}")
    print("Combinable hyperparameters:")
    for key, value in config["combinable"].items():
        print(f"{key}: {value}")
    print("==============================================")


def main(args):
    # Load hyperparameters from the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs', args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    print_hyperparameters(config)

    # Extract combinable and fixed hyperparameters
    combinable = config["combinable"]
    fixed = config["fixed"]

    combinable_keys, combinable_values = zip(*combinable.items())
    combinations = [dict(zip(combinable_keys, v)) for v in itertools.product(*combinable_values)]

    fixed_keys, fixed_values = zip(*fixed.items())
    fixed_combinations = [dict(zip(fixed_keys, v)) for v in itertools.product(*fixed_values)]

    directories = [
        r"C:\Users\piete\OneDrive - JKU\OneDrive - Johannes Kepler Universität Linz\MSc in AI - JKU\Courses\Thesis\Repositories\MT_Normalizer_free_Neural_Nets"
    ]

    total_combinations = len(combinations) * len(fixed_combinations) * len(directories)
    counter = 1
    for directory in directories:
        print("Training in directory:", directory)
        os.chdir(directory)

        # Iterate over all combinations of hyperparameters
        for combo in combinations:
            for fixed_combo in fixed_combinations:
                combined = {**fixed_combo, **combo}
                print("==============================================")
                print("Using hyperparameters:", combined)
                print(f"Training combination {counter} of {total_combinations}")
                print("==============================================")
                
                # make the config file for a single run
                combo_config_name = f"combo_config_{counter}.json"
                combo_config_path = os.path.join(script_dir, 'configs', combo_config_name)
                with open(combo_config_path, 'w') as f:
                    json.dump(combined, f)
                
                # Construct the command with the path to the temporary JSON file
                command = f"python run_training.py --config {combo_config_name}"
                
                subprocess.run(command, shell=True)
                gc.collect()
                os.remove(combo_config_path)
                counter += 1
    
    print_hyperparameters(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search script parameters")
    parser.add_argument('--config', type=str, default="config_parameter_search.json", help='Configuration file name')
    args = parser.parse_args()

    main(args)