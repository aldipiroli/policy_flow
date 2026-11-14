from pathlib import Path
from unittest.mock import MagicMock

from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig

from policy_flow.utils.misc import find_all_cfg, load_config


def disable_pretrained(node):
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            if key == "pretrained":
                node[key] = False
            else:
                disable_pretrained(node[key])
    elif isinstance(node, ListConfig):
        for item in node:
            disable_pretrained(item)


def test_model_loading():
    folder_path = Path("policy_flow/configs/models")
    configs = find_all_cfg(folder_path)
    for curr_config in configs:
        cfg = load_config(config_file=curr_config)
        disable_pretrained(cfg)  # no weight download in ci
        get_class(cfg._target_)(cfg, MagicMock())
