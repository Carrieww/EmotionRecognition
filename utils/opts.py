import argparse
import yaml

class Config:
    """dict -> Class"""
    def __init__(self, entries: dict={}):
        for k, v in entries.items():
            if k != 'params' and isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v


def load_config(file_path: str) -> dict:
    """
    Load config from YAML
    Args:
        file_path (str): path
    Returns:
        config (dict): config
    """
    f = open(file_path, 'r', encoding = 'utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def parse_opt():
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        '--config',
        type = str,
        default = 'Configs/Ravdess.yaml',
        help = 'path to the configuration file (yaml)'
    )
    args = parser.parse_args()
    config_dict = load_config(args.config)
    config = Config(config_dict)

    return config