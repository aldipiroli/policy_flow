import pytest
import torch
from omegaconf import OmegaConf

from policy_flow.utils.misc import (
    check_timeframe_validity,
    disable_pretrained_in_cfg,
    find_all_cfg,
    load_config,
    make_artifacts_dirs,
)
from policy_flow.utils.ml import count_model_params


def test_load_config():
    cfg = load_config(config_file="policy_flow/configs/artifacts/default.yaml")
    assert cfg is not None


def test_make_artifacts_dirs(tmp_path):
    cfg = load_config(config_file="policy_flow/configs/artifacts/default.yaml")
    cfg = OmegaConf.create({"artifacts": cfg})
    cfg.artifacts.root_dir = tmp_path
    make_artifacts_dirs(cfg)
    for name in cfg.artifacts.folders:
        assert cfg.artifacts.folders[name].is_dir()


def test_count_model_params():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    n_param, n_trainable, n_frozen = count_model_params(model)
    assert n_param == (10 * 5 + 5) + (5 * 2 + 2)
    assert n_trainable == n_param
    assert n_frozen == 0

    for p in model[0].parameters():
        p.requires_grad = False
    n_param, n_trainable, n_frozen = count_model_params(model)
    assert n_param == (10 * 5 + 5) + (5 * 2 + 2)
    assert n_trainable == 5 * 2 + 2
    assert n_frozen == 10 * 5 + 5


def test_find_all_cfg(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1")
    (tmp_path / "b.yml").write_text("x: 2")
    result = find_all_cfg(tmp_path)
    assert sorted(p.name for p in result) == ["a.yaml", "b.yml"]


def test_disable_pretrained_in_cfg():
    cfg = OmegaConf.create(
        {
            "pretrained": True,
            "model": {"pretrained": True, "lr": 0.1},
            "layers": [{"pretrained": True}],
        }
    )
    disable_pretrained_in_cfg(cfg)
    assert cfg.pretrained is False
    assert cfg.model.pretrained is False
    assert cfg.layers[0].pretrained is False


@pytest.mark.parametrize(
    "tf, expected",
    [
        (("2024-01-01", "2024-02-01"), True),
        (("2024-02-01", "2024-01-01"), False),
        (("2024-01-01", "2024-01-01"), False),
        (("2024-02-30", "2024-03-01"), False),
        (("2024-01-01", "2024-13-01"), False),
        (("2024-02-29", "2024-03-01"), True),
        (("2023-02-29", "2023-03-01"), False),
        (("2024-1-5", "2024-02-01"), False),
    ],
)
def test_check_timeframe_validity(tf, expected):
    assert check_timeframe_validity(tf) == expected
