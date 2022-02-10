import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List

class WarmupLinearDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_factor: float = 0.001,
        warmup_iters: int = 10,
        warmup_method: str = "linear",
        end_epoch: int = 300,
        final_lr_factor: float = 0.003,
        last_epoch: int = -1,
    ):
        """
        Multi Step LR with warmup

        Args:
            optimizer (torch.optim.Optimizer): optimizer used.
            warmup_factor (float): lr = warmup_factor * base_lr
            warmup_iters (int): iters to warmup
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch(int):  The index of last epoch. Default: -1.
        """
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.end_epoch = end_epoch
        assert 0 < final_lr_factor < 1
        self.final_lr_factor = final_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        linear_decay_factor = _get_lr_linear_decay_factor_at_iter(
            self.last_epoch, self.warmup_iters, self.end_epoch, self.final_lr_factor)
        return [
            base_lr * warmup_factor * linear_decay_factor for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        linear_decay_factor = _get_lr_linear_decay_factor_at_iter(
            self.last_epoch, self.warmup_iters, self.end_epoch, self.final_lr_factor)
        return [
            base_lr * warmup_factor * linear_decay_factor for base_lr in self.base_lrs
        ]


def _get_lr_linear_decay_factor_at_iter(iter: int, start_epoch: int, end_epoch: int,
                                        final_lr_factor: float):
    assert iter <= end_epoch
    if iter <= start_epoch:
        return 1.0
    alpha = (iter - start_epoch) / (end_epoch - start_epoch)
    lr_step = final_lr_factor * alpha + 1 - alpha

    return lr_step


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int,
                               warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    elif method == "burnin":
        return (iter / warmup_iters)**4
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
