import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, cfg, logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.logger.info(f"Loading model {type(self).__name__}")
