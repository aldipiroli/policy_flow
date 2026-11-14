import argparse

from hydra.utils import get_class

import policy_flow.utils.misc as misc_utils


def train(args):
    cfg, logger = misc_utils.run_initial_setup(args, prefix="eval")
    trainer = get_class(cfg.optim._target_)(cfg, logger)
    trainer.initialize()
    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)
    trainer.evaluate_env()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str, default="cfg/cfg.yaml", help="cfg path")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    train(args)
