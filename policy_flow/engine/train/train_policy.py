import gymnasium as gym
import torch
from tqdm import tqdm

from policy_flow.engine.train.trainer_base import TrainerBase


class TrainPolicy(TrainerBase):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="human")

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.optim.epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_and_save(epoch)
        self.env.close()
        self.exp_logger.close()

    def evaluate_and_save(self, epoch):
        if epoch % 100 == 0:
            self.evaluate_one_epoch()
            self.save_checkpoint()
            self.model.train()

    def train_one_epoch(self, mode="train"):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=mode)
        for _batch_idx, (state, actions) in pbar:
            with self.autocast():
                state = state.to(self.device)
                actions = actions.to(self.device)
                pred = self.model(state)
            loss, loss_dict = self.loss_fn(pred, actions)
            self.exp_logger.write_dict(loss_dict, self.total_iters_train, prefix=mode)
            self.optimize_step(loss)
            self.update_pbar(pbar, mode, loss)
            self.exp_logger.write_float(self.get_lr(), self.total_iters_train, f"{mode}/lr")
            self.total_iters_train += 1
        self.exp_logger.write_float(self.get_epoch_time(), self.epoch, f"{mode}/epoch_time")

    @torch.no_grad()
    def evaluate_one_epoch(self, mode="val"):
        self.model.eval()
        self.metrics.reset()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=mode)
        for _batch_idx, (state, actions) in pbar:
            with self.autocast():
                state = state.to(self.device)
                actions = actions.to(self.device)
                pred = self.model(state)
            self.metrics.update(pred, actions)
            pbar.set_postfix({"mode": mode})
        results = self.metrics.compute()
        self.metrics.print_metrics()
        self.exp_logger.write_dict(results, self.total_iters_train, prefix=mode)
        self.ckpt_manager.is_best_ckpt(results[self.metrics.main_metric])

    def post_process_actions(self, pred):
        pred = pred * 512.0
        pred = pred.reshape(-1, self.cfg.dataset.action_size)
        return pred

    @torch.no_grad()
    def evaluate_env(self):
        self.model.eval()
        state, info = self.env.reset()
        done = False
        max_coverage = 0.0

        while not done:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            state = self.train_dataset.normalize_state(state)
            state = state.unsqueeze(0)

            pred = self.model(state)
            actions = self.post_process_actions(pred)

            for curr_action in actions:
                curr_action = curr_action.cpu().numpy().clip(0, 512)
                state, reward, terminated, truncated, info = self.env.step(curr_action)
                self.env.render()

            max_coverage = max(max_coverage, info["coverage"])
            done = terminated or truncated

        self.logger.info(f"Success: {info['is_success']}, max coverage: {max_coverage:.4f}")
