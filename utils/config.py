import yaml
from pathlib import Path
import argparse


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def get_config_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    config = load_config(args.config)

    for key, value in vars(args).items():
        if value is not None and not hasattr(config, key):
            setattr(config, key, value)

    return config, args
