import torch.nn as nn

from policy_flow.models.base_model import BaseModel


class FlowPolicy(BaseModel):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        input_size = cfg.state_size + (cfg.chunk_size * cfg.action_size) + 1
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, cfg.chunk_size * cfg.action_size),
        )

    def forward(self, x):
        b = x.shape[0]
        actions = self.model(x)
        actions = actions.reshape(b, -1)
        return actions
