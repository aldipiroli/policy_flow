import torch
import torch.nn as nn

from policy_flow.models.base_model import BaseModel


class TransformerFlowPolicy(BaseModel):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        self.state_size = cfg.state_size
        self.action_size = cfg.action_size
        self.chunk_size = cfg.chunk_size
        self.dim = cfg.dim
        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.project_input = nn.Linear(self.state_size + self.action_size + 1, self.dim)
        self.project_output = nn.Linear(self.dim, self.action_size)

    def forward(self, x):
        b = x.shape[0]
        state = x[:, : self.state_size]
        actions = x[:, self.state_size : self.state_size + (self.action_size * self.chunk_size)]
        actions = actions.reshape(b, self.chunk_size, -1)
        time = x[:, -1].unsqueeze(-1)

        state = state[:, None, :].expand(-1, actions.shape[1], -1)
        time = time[:, None, :].expand(-1, actions.shape[1], -1)

        xin = torch.cat([state, actions, time], -1)
        xin = self.project_input(xin)
        xout = self.transformer(xin)
        xout = self.project_output(xout)
        pred_actions = xout.reshape(b, self.action_size * self.chunk_size)
        return pred_actions
