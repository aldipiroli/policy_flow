import torch.nn.functional as F
from torchmetrics import MeanMetric

from policy_flow.engine.metrics.base_metrics import BaseMetrics


class MSEMetric(BaseMetrics):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.mse = MeanMetric()
        self.set_main_metric_name(name="mse")

    def update(self, preds, targets):
        mse = F.mse_loss(preds, targets)
        self.mse.update(mse.detach().cpu())

    def compute(self):
        mse = self.mse.compute().item()
        return {"mse": mse}

    def reset(self):
        self.mse.reset()
