from pathlib import Path

import torch
from aim import Image, Run, Text
from omegaconf import OmegaConf


class ExperimentLogger:
    def __init__(self, cfg):
        self.log_every = cfg.optim.experiment_log_frequency

        aim_folder = Path(cfg.artifacts.folders["aim_log"])
        repo_path = aim_folder.parent.parent
        exp_name = aim_folder.parent.name
        self.run = Run(repo=str(repo_path), experiment=exp_name)

        params = OmegaConf.to_container(cfg, resolve=True)
        params.pop("artifacts", None)
        for key, value in params.items():
            self.run[key] = value

    def close(self):
        self.run.close()

    def write_float(self, value, step, name):
        if step % self.log_every != 0:
            return
        self.run.track(_float(value), name=name, step=step)

    def write_dict(self, values, step, prefix=""):
        if step % self.log_every != 0:
            return
        for k, v in values.items():
            name = f"{prefix}/{k}" if prefix else k
            self.run.track(_float(v), name=name, step=step)

    def write_images(self, img, step, prefix="img"):
        if step % self.log_every != 0:
            return
        self.run.track(Image(img), name=prefix, step=step)

    def write_text(self, text, name, step):
        if step % self.log_every != 0:
            return
        self.run.track(Text(str(text)), name=name, step=step)


def _float(v):
    return v.item() if isinstance(v, torch.Tensor) else float(v)
