import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config_file, overrides=None):
    config_file = Path(config_file).resolve()
    config_dir = config_file.parent
    config_name = config_file.stem

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        return compose(config_name=config_name, overrides=overrides or [])


def make_artifacts_dirs(cfg, debug=False, prefix=""):
    root = Path(cfg.artifacts.root_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_prefix = f"{prefix}_" if prefix else ""
    if debug:
        root = root / f"{name_prefix}debug"
        if root.exists():
            shutil.rmtree(root)
    else:
        root = root / f"{name_prefix}{stamp}"

    cfg.artifacts.root_dir = root
    root.mkdir(parents=True, exist_ok=True)
    for name in cfg.artifacts.folders:
        path = root / cfg.artifacts.folders[name]
        path.mkdir(parents=True, exist_ok=True)
        cfg.artifacts.folders[name] = path
    return cfg


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def dump_config(cfg, logger):
    cfg_dump_path = Path(cfg.artifacts.root_dir) / "cfg.yaml"
    OmegaConf.save(cfg, cfg_dump_path, resolve=True)
    logger.info(cfg)


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_initial_setup(args, prefix=""):
    overrides = getattr(args, "overrides", [])
    cfg = load_config(args.cfg_file, overrides=overrides)
    set_random_seeds(cfg.get("seed", 42))
    cfg = make_artifacts_dirs(cfg, debug=args.debug, prefix=prefix)
    logger = get_logger(cfg.artifacts.folders["log"])
    if not args.debug:
        save_git_state(cfg, logger)
    dump_config(cfg, logger)
    return cfg, logger


def find_all_cfg(folder_path):
    folder_path = Path(folder_path)
    assert folder_path.is_dir(), f"Path does not exist: {folder_path}"
    return [p for p in folder_path.rglob("*") if p.suffix in (".yaml", ".yml") and p.is_file()]


def disable_pretrained_in_cfg(node):
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            if key == "pretrained":
                node[key] = False
            else:
                disable_pretrained_in_cfg(node[key])
    elif isinstance(node, ListConfig):
        for item in node:
            disable_pretrained_in_cfg(item)


def save_git_state(cfg, logger):
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if not commit:
        logger.warning("Not a git repo — code state not saved.")
        return None

    dirty = bool(git("status", "--porcelain"))
    state = {
        "commit": commit,
        "dirty": dirty,
        "cmd": " ".join(sys.argv),
    }
    if dirty:
        git("add", "-N", ".")
        patch = git("diff", "HEAD")
        git("reset")
        (Path(cfg.artifacts.root_dir) / "uncommitted.patch").write_text(patch)
        state["patch"] = "uncommitted.patch"

    (Path(cfg.artifacts.root_dir) / "git.json").write_text(json.dumps(state, indent=2))
    logger.info(f"Code state: {commit[:12]}{' (dirty)' if dirty else ''}")
    return commit


def is_valid_date(s):
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        return False
    year, month, day = map(int, m.groups())
    return year >= 1 and 1 <= month <= 12 and 1 <= day <= monthrange(year, month)[1]


def check_timeframe_validity(tf):
    start, end = tf
    return is_valid_date(start) and is_valid_date(end) and start < end
