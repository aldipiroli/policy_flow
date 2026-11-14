import torch


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def to_device(x):
    x = x.to(get_device())
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def count_model_params(model, logger=None):
    trainable = 0
    frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    n_param = trainable + frozen
    if logger is not None:
        logger.info(f"Num Parameters: {n_param / 1e6} M")
        logger.info(
            f"Trainable: {trainable / 1e6} M ({100 * trainable / n_param:.2f}%), "
            f"Frozen: {frozen / 1e6} M ({100 * frozen / n_param:.2f}%)"
        )
    return n_param, trainable, frozen


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]
