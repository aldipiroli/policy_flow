from pathlib import Path
from unittest.mock import MagicMock

from hydra.utils import get_class

from policy_flow.utils.misc import find_all_cfg, load_config


def test_loss_loading():
    folder_path = Path("policy_flow/configs/losses")
    configs = find_all_cfg(folder_path)
    for curr_config in configs:
        cfg = load_config(config_file=curr_config)
        get_class(cfg._target_)(cfg, MagicMock())
