import argparse

import gymnasium as gym
import numpy as np
from hydra.utils import get_class

import policy_flow.utils.misc as misc_utils


def evaluate(args):
    cfg, logger = misc_utils.run_initial_setup(args, prefix="eval")
    trainer = get_class(cfg.optim._target_)(cfg, logger)
    trainer.initialize()
    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)
    if args.multiple_evals:
        results = trainer.evaluate_envs()
        seed = results["best_seed"]
        logger.info(f"Replaying best seed: {seed}")
    else:
        seed = int(np.random.default_rng().integers(100))
        logger.info(f"Evaluating random seed: {seed}")
    trainer.env.close()
    render_single_seed(trainer, seed, logger)


def render_single_seed(trainer, seed, logger):
    video_folder = str(trainer.cfg.artifacts.folders.img)
    trainer.env = gym.wrappers.HumanRendering(
        gym.wrappers.RecordVideo(
            gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array"),
            video_folder=video_folder,
            name_prefix=f"seed-{seed}",
            episode_trigger=lambda _: True,
        )
    )
    result = trainer.evaluate_env(seed, render=True)
    logger.info(result)
    trainer.env.close()
    logger.info(f"Saved replay video to {video_folder}/seed-{seed}-episode-0.mp4")
    trainer.exp_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str, default="cfg/cfg.yaml", help="cfg path")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--multiple_evals", action="store_true")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    evaluate(args)
