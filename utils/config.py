
import yaml
import argparse
from pathlib import Path

import utils


def get_config(default_config_path):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default=default_config_path)

    # This line is key to pull the config file (if specified)
    temp_args, _ = parser.parse_known_args()

    with open(temp_args.config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create argparse arguments for all variables in the config yaml
    # 2 level dictionary config is supported, subkeys will be presented as --key.subkey args
    # Motivation:
    #   Hyperopt frameworks can pass hyperparameters easily to the training
    #   2 level dictionaries can be passed as a top level dict or by just passing a value for --key.subkey
    # How this is implemented: 
    #   The default for 2 level dict keys is "load-from-child-args" and in a laterstage these dictionaries 
    #   will be reconstructed from --key.subkey args
    # Examples:
    #   --scheduler {'gamma': 0.1, 'step_size': 50, 'type': 'torch.optim.lr_scheduler.StepLR'}
    #   --optimizer.lr 5.0e-06
    #   In the later case optimizer.lr is overwritten, other subkeys are taken from the yaml.
    for k, v in config.items():
        if isinstance(v, dict):
            arg(f"--{k}", default="load-from-child-args", type=yaml.safe_load, help=" ")
            for k2, v2 in v.items():
                arg(f"--{k}.{k2}", default=v2, type=type(v2), help=" ")
        else:
            arg(f"--{k}", default=v, type=type(v), help=" ")

    args = parser.parse_args()
    flat_config = args.__dict__
    
    # Process 2 level dictionary hyperparameters: rebuild dictionary configs from subkey values
    # If an argument's value is "load-from-child-args", we search for subkeys, and construct a dictionary from them
    # Key.Subkey itmes are then are dropped from the dictionary.
    config = {}
    for k, v in flat_config.items():
        if v == "load-from-child-args":
            config[k] = {}
            # Search for subkeys
            for k2, v2 in flat_config.items():
                if k2.startswith(f"{k}."):
                    config[k][k2.replace(f"{k}.", "")] = v2
            flat_config[k] = config[k]
        else: 
            config[k] = v
    
    
    flat_config = {k.replace(".","_"): v for k, v in flat_config.items()}
    config['experiment_name'] = config['experiment_name'].format(**flat_config)
    # config['experiment_name'] = config['experiment_name'].split('/')[-1] # For the Pretraining epochs sweep
    print(utils.pretty_print(config))
    return config