from abc import abstractmethod

import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger

    @abstractmethod
    def forward(self, preds, labels):
        pass
