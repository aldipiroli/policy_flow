import math
import time
from abc import ABC, abstractmethod

import torch
from hydra.utils import get_class
from torch.utils.data import DataLoader

from policy_flow.utils import ml as ml_utils
from policy_flow.utils.ckpt_handler import CheckpointManager
from policy_flow.utils.exp_logger import ExperimentLogger


class TrainerBase(ABC):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.exp_logger = self.set_exp_logger()

        self.device = ml_utils.get_device()
        self.logger.info(f"Using device: {self.device}")
        self.ckpt_manager = CheckpointManager(cfg, logger)
        self.set_variables()
        self.set_mixed_precision()

    def initialize(self):
        # datasets
        train_dataset = get_class(self.cfg.dataset._target_)(cfg=self.cfg.dataset, split="train", logger=self.logger)
        val_dataset = get_class(self.cfg.dataset._target_)(cfg=self.cfg.dataset, split="val", logger=self.logger)
        self.set_dataset(train_dataset, val_dataset)

        # models
        model = get_class(self.cfg.model._target_)(self.cfg.model, self.logger)
        self.set_models(model)

        # loss & optim
        loss = get_class(self.cfg.loss._target_)(self.cfg.loss, self.logger)
        self.set_optimizer()
        self.set_loss_function(loss)

        # metrics
        self.metrics = get_class(self.cfg.metrics._target_)(self.cfg.metrics, self.logger)

    def set_variables(self):
        self.epoch = 0
        self.total_iters_train = 0
        self.epoch_start_time = time.time()

    def set_mixed_precision(self):
        self.use_amp = self.cfg.optim.mixed_precision
        self.amp_dtype = getattr(torch, self.cfg.optim.amp_dtype)
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=use_scaler)
        self.logger.info(f"Mixed precision: {self.use_amp}, dtype: {self.amp_dtype}")

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)

    def set_exp_logger(self):
        return ExperimentLogger(self.cfg)

    def save_checkpoint(self):
        self.ckpt_manager.save(self.model, self.optimizer, self.epoch, self.total_iters_train)

    def load_latest_checkpoint(self, ckpt_dir):
        state = self.ckpt_manager.load_latest_ckpt(ckpt_dir, self.model, self.optimizer)
        self.epoch = state["epoch"]
        self.total_iters_train = state["total_iters"]

    def load_checkpoint(self, ckpt_path):
        state = self.ckpt_manager.load_ckpt(ckpt_path, self.model, self.optimizer)
        self.epoch = state["epoch"]
        self.total_iters_train = state["total_iters"]

    def set_dataset(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.optim.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.optim.train_num_workers,
            prefetch_factor=1,
            persistent_workers=False,
        )
        self.logger.info(f"Train Dataset: {self.train_dataset}")

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.optim.validation_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.optim.val_num_workers,
            persistent_workers=False,
        )
        self.logger.info(f"Val Dataset: {self.val_dataset}")

    def set_optimizer(self):
        optimizer = self.cfg.optim.optimizer
        lr = self.cfg.optim.lr
        wd = self.cfg.optim.weight_decay

        optimizers = {
            "AdamW": lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd),
            "Adam": lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0),
            "SGD": lambda: torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0),
        }
        assert optimizer in optimizers
        self.optimizer = optimizers[optimizer]()
        self.logger.info(f"Optimizer: {self.optimizer}")

    def lr_at(self, step):
        optim = self.cfg.optim
        total_steps = optim.epochs * len(self.train_loader)
        warmup = min(1.0, step / max(1, optim.warmup_steps))
        return optim.lr * warmup * 0.5 * (1 + math.cos(math.pi * step / total_steps))

    def set_models(self, model):
        self.model = model
        self.model = self.model.to(self.device)

        self.logger.info("Model:")
        ml_utils.count_model_params(self.model, self.logger)

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)
        self.logger.info(f"Loss function {self.loss_fn}")

    def update_pbar(self, pbar, mode, loss=None):
        pbar.set_postfix(
            {
                "mode": mode,
                "epoch": f"{self.epoch}/{self.cfg.optim.epochs}",
                "loss": loss.item() if loss is not None else None,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        )

    def optimize_step(self, loss):
        accum_steps = self.cfg.optim.grad_accum_steps
        scaled_loss = loss / accum_steps
        self.scaler.scale(scaled_loss).backward()

        is_update_step = (self.total_iters_train + 1) % accum_steps == 0
        if is_update_step:
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_at(self.total_iters_train)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    def get_epoch_time(self):
        now = time.time()
        epoch_time = now - self.epoch_start_time
        self.epoch_start_time = now
        return epoch_time

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.optim.epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_one_epoch()
            self.save_checkpoint()
        self.exp_logger.close()

    @abstractmethod
    def train_one_epoch(self):
        pass

    @abstractmethod
    def evaluate_one_epoch(self):
        pass
