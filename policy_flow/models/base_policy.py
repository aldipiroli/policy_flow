import torch
import torch.nn as nn
import torch.nn.functional as F

from policy_flow.models.base_model import BaseModel


class BasePolicy(BaseModel):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.model = nn.Sequential(
            nn.Linear(cfg.state_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, cfg.chunk_size * cfg.action_size),
        )

    def forward(self, x):
        actions = self.model(x)
        return actions
