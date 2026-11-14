from pathlib import Path
from unittest.mock import MagicMock

from hydra.utils import get_class

from policy_flow.utils import misc as misc_utils
from policy_flow.utils.misc import disable_pretrained_in_cfg, find_all_cfg, load_config


def test_trainer_initialization(tmp_path):
    folder_path = Path("policy_flow/configs")
    configs = find_all_cfg(folder_path)
    for curr_config in configs:
        if not curr_config.name.startswith("train_"):
            continue
        cfg = load_config(config_file=curr_config)
        cfg.artifacts.root_dir = tmp_path
        cfg = misc_utils.make_artifacts_dirs(cfg)
        disable_pretrained_in_cfg(cfg)  # no weight download in ci
        trainer = get_class(cfg.optim._target_)(cfg, MagicMock())
        trainer.initialize()
