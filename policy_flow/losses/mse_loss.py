import torch.nn as nn

from policy_flow.losses.base_loss import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, preds, targets):
        mse_loss = self.mse_loss_fn(preds, targets)
        loss = mse_loss
        loss_dict = {"mse": mse_loss}
        return loss, loss_dict
