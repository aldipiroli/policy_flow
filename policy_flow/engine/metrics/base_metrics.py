import torch.nn as nn


class BaseMetrics(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.main_metric = None

    def print_metrics(self):
        metrics = self.compute()
        self.logger.info(",".join(metrics.keys()))
        self.logger.info(",".join(f"{v:.4f}" for v in metrics.values()))

    def set_main_metric_name(self, name):
        self.main_metric = name
