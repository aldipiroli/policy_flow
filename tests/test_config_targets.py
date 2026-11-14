from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf

from policy_flow.utils.misc import find_all_cfg


def test_config_targets_resolve():
    for curr_config in find_all_cfg(Path("policy_flow/configs")):
        cfg = OmegaConf.load(curr_config)
        if "_target_" in cfg:
            get_class(cfg._target_)
