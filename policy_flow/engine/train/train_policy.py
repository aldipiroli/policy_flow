import torch
from tqdm import tqdm

from shiko.engine.train.trainer_base import TrainerBase
from shiko.utils import ml as ml_utils


class TrainPushT(TrainerBase):
    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.optim.epochs):
            self.epoch = epoch
            self.train_one_epoch()
        self.exp_logger.close()

    def evaluate_and_save(self):
        if self.total_iters_train % self.cfg.optim.steps_per_epoch != 0:
            return
        self.evaluate_one_epoch()
        self.save_checkpoint()
        self.model.train()

    def train_one_epoch(self, mode="train"):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=mode)
        for _batch_idx, data in pbar:
            with self.autocast():
                pred, target, valid = self.forward_batch(data)
            loss, loss_dict = self.loss_fn(pred, target, valid)
            self.exp_logger.write_dict(loss_dict, self.total_iters_train, prefix=mode)
            self.optimize_step(loss)
            self.update_pbar(pbar, mode, loss)
            self.exp_logger.write_float(self.get_lr(), self.total_iters_train, f"{mode}/lr")
            self.total_iters_train += 1
            self.evaluate_and_save()
        self.exp_logger.write_float(self.get_epoch_time(), self.epoch, f"{mode}/epoch_time")

    @torch.no_grad()
    def evaluate_one_epoch(self, mode="val"):
        self.model.eval()
        self.metrics.reset()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=mode)
        for batch_idx, data in pbar:
            with self.autocast():
                pred, target, valid = self.forward_batch(data)
            self.metrics.update(pred, target, valid)
            pbar.set_postfix({"mode": mode})
            if batch_idx >= self.cfg.optim.eval_steps:
                break
        results = self.metrics.compute()
        self.metrics.print_metrics()
        self.exp_logger.write_dict(results, self.total_iters_train, prefix=mode)
        self.ckpt_manager.is_best_ckpt(results[self.metrics.main_metric])
