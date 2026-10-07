import math


def annealing(
    t: int,
    warm_up_interval: int = 10000,
    scheme: str = 'linear',
    anneal: bool = False,
) -> float:
    """
    warmup the log probability during training, linearly from 0.1 to 1.0 over warm_up_interval steps.
    
    Args:
        t (int): current training step
        warm_up_interval (int): number of steps to warm up
        scheme (str): annealing scheme to use ('linear' or 'sigmoid')
        anneal (bool): whether to apply annealing
    Returns:
        float: annealing factor
    """
    if not anneal:
        return 1.0
    else:
        progress = min(1.0, t / warm_up_interval)
        if scheme == 'linear':
            return 0.1 + 0.9 * progress  # linear annealing
        elif scheme == 'sigmoid':
            return 0.1 + 0.9 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
        else:
            raise ValueError(f"Unknown annealing scheme: {scheme}")
