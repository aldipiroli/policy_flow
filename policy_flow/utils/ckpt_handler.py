from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.ckpt_dir = Path(cfg.artifacts.folders["ckpts"])
        self.logger = logger
        self.set_variables()

    def set_variables(self):
        self.best_metric_value = None
        self.best_ckpt = False

    def save(self, model, optimizer, epoch, total_iters):
        if (epoch + 1) % self.cfg.optim.save_ckpt_every != 0:
            return

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_iters": total_iters,
        }

        last_path = self.ckpt_dir / "last.pt"
        torch.save(checkpoint, last_path)
        self.logger.info(f"Saved checkpoint in: {last_path}")

        if not self.cfg.optim.save_only_best:
            path = self.ckpt_dir / f"ckpt_{epoch:04d}.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved checkpoint in: {path}")

        if self.best_ckpt:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"Saved checkpoint in: {best_path}, best_ckpt: {self.best_ckpt}, best_metric: {self.best_metric_value}"
            )

    def latest(self, ckpt_dir):
        if not ckpt_dir.exists():
            return None
        ckpts = ckpt_dir.glob("ckpt_*.pt")
        return max(ckpts, key=lambda p: int(p.stem.split("_")[1]), default=None)

    def load(self, path, model, optimizer=None):
        self.logger.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return {"epoch": ckpt.get("epoch", 0), "total_iters": ckpt.get("total_iters", 0)}

    def load_latest_ckpt(self, ckpt_dir, model, optimizer=None):
        path = self.latest(ckpt_dir)
        if path is None:
            self.logger.info("No checkpoints found.")
            return None
        return self.load(path, model, optimizer)

    def load_ckpt(self, ckpt_path, model, optimizer=None):
        return self.load(ckpt_path, model, optimizer)

    def is_best_ckpt(self, curr_metric_value):
        best = self.best_metric_value
        if self.cfg.metrics.higher_is_better:
            self.best_ckpt = best is None or curr_metric_value > best
        else:
            self.best_ckpt = best is None or curr_metric_value < best
        if self.best_ckpt:
            self.best_metric_value = curr_metric_value
