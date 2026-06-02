from typing import Union

import numpy as np
import torch
# from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LRScheduler
# from torch.optim.lr_scheduler import StepLR
# from typeguard import check_argument_types


class WarmupLR(LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 5000,
            min_lr=2e-5,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lrs = []
        for lr in self.base_lrs:
            if self.warmup_steps == 0:
                lr = lr * step_num ** -0.5
                if lr < self.min_lr:
                    lr = self.min_lr
            else:
                lr = lr * self.warmup_steps ** 0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
                if lr < self.min_lr and step_num > self.warmup_steps:
                    lr = self.min_lr
            lrs.append(lr)
        return lrs

    def set_step(self, step: int):
        self.last_epoch = step


class WarmupDecayingCosineAnnealingWarmRestarts(LRScheduler):
    """Warmup + decaying-peak SGDR learning-rate scheduler.

    This is cosine annealing with warm restarts (SGDR, Stochastic Gradient
    Descent with warm Restarts) in which the upper and lower bounds decay at
    every restart. After each restart the learning rate returns to a peak that
    is lower than the previous one, so the schedule keeps oscillating while its
    overall envelope decreases toward convergence.

    The learning rate has two phases (see :meth:`ctxadjust_lr` for the actual
    computation):

    1. Warmup phase (``step < ws``):
        Linear ramp from 0 up to ``eta_max``::

            lr = step * (eta_max / ws)

    2. Cosine annealing with periodic warm restarts (``step >= ws``):
        Each cycle lasts ``T_0`` steps; within a cycle the learning rate
        follows a cosine curve smoothly decreasing from the upper bound to the
        lower bound, then warm-restarts back to the upper bound at the end of
        the cycle. In addition, the bounds of the ``k``-th cycle are both
        scaled by the decay factor ``tmctx ** k`` so that the peak of each
        cycle decreases over time::

            T_cur = (step + ws) % T_0      # relative step within the cycle
            k     = (step + ws) // T_0     # index of the current cycle
            lo    = eta_min * tmctx ** k   # lower bound for this cycle
            hi    = eta_max * tmctx ** k   # upper bound for this cycle
            lr    = lo + 0.5 * (hi - lo) * (1 + cos(pi * T_cur / T_0))

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Only used in :meth:`get_lr` to choose a branch; both
            branches currently call :meth:`ctxadjust_lr`, so behavior is the
            same either way.
        min_lr: Lower bound on the learning rate (kept for compatibility; not
            applied in the annealing formula).
        last_epoch: Index of the previous step; ``-1`` starts from scratch.
        T_0: Number of steps in a single cosine annealing cycle (the interval
            between two warm restarts).
        eta_max: Initial upper bound of the cosine annealing (peak lr).
        eta_min: Initial lower bound of the cosine annealing (trough lr).
        T_mul / T_mult: Reserved fields; currently unused.

    Note:
        :meth:`get_lr` calls :meth:`ctxadjust_lr` without arguments, so the
        defaults of :meth:`ctxadjust_lr` (``T_0=15000, eta_min=6e-5,
        eta_max=9e-5, tmctx=0.98, ws=5000``) are used instead of the values
        passed to ``__init__``. To make the constructor arguments take effect,
        pass them explicitly to :meth:`ctxadjust_lr`.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 25000,
            min_lr=1e-5,
            last_epoch: int = -1,
            T_0=1500,
            eta_max=0.1,
            eta_min=0.,
            T_mul=2,
            T_mult=0.9999,
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.eta_min = eta_min
        self.T_0 = T_0
        self.eta_max = eta_max
        self.T_mul = T_mul
        self.T_mult = T_mult
        super().__init__(optimizer, last_epoch)

    def ctxadjust_lr(self, T_0=15000, eta_min=0.00006, eta_max=0.00009, tmctx=0.98, ws=5000):
        step_num = self.last_epoch + 1
        T_cur = (step_num + ws) % T_0
        T_i = T_0
        T_curX = (step_num + ws) // T_0
        cur_lr = eta_min * (tmctx ** T_curX) + 0.5 * (
                eta_max * (tmctx ** T_curX) - eta_min * (tmctx ** T_curX)
        ) * (1 + np.cos(np.pi * T_cur / T_i))
        if ws > step_num:
            cur_lr = step_num * (eta_max / ws)
        return cur_lr

    def get_lr(self):
        lrs = []
        for _ in self.base_lrs:
            lrs.append(self.ctxadjust_lr())
        return lrs

    def set_step(self, step: int):
        self.last_epoch = step

