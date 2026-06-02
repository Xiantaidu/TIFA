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
        Within a cycle the learning rate follows a cosine curve smoothly
        decreasing from the upper bound to the lower bound, then warm-restarts
        back to the upper bound at the end of the cycle. The bounds of the
        ``k``-th cycle are both scaled by the decay factor ``tmctx ** k`` so
        that the peak of each cycle decreases over time. Two cycle-length
        schemes are available, selected by ``T_mul``:

        * ``T_mul == 1`` (default): fixed-length cycles, every cycle lasts
          ``T_0`` steps::

            T_cur = (step + ws) % T_0
            k     = (step + ws) // T_0
            T_i   = T_0

        * ``T_mul == 2``: geometrically growing cycles, the ``k``-th cycle
          lasts ``T_0 * T_mult ** k`` steps::

            k     = floor(log(step * (T_mult - 1) / T_0 + 1) / log(T_mult))
            T_cur = step - T_0 * (T_mult ** k - 1) / (T_mult - 1)
            T_i   = T_0 * T_mult ** k

        In both cases the learning rate is::

            lo = eta_min * tmctx ** k
            hi = eta_max * tmctx ** k
            lr = lo + 0.5 * (hi - lo) * (1 + cos(pi * T_cur / T_i))

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Only used in :meth:`get_lr` to choose a branch; both
            branches currently call :meth:`ctxadjust_lr`, so behavior is the
            same either way.
        min_lr: Lower bound on the learning rate (kept for compatibility; not
            applied in the annealing formula).
        last_epoch: Index of the previous step; ``-1`` starts from scratch.
        T_0: Number of steps in the first cosine annealing cycle.
        eta_max: Initial upper bound of the cosine annealing (peak lr).
        eta_min: Initial lower bound of the cosine annealing (trough lr).
        T_mul: Cycle-length scheme forwarded to :meth:`ctxadjust_lr` by
            :meth:`get_lr`: ``1`` for fixed-length cycles, ``2`` for
            geometrically growing cycles.
        T_mult: Cycle-length growth factor used when ``T_mul == 2`` (e.g.
            ``2.0`` doubles the cycle length after each restart). Forwarded to
            :meth:`ctxadjust_lr` by :meth:`get_lr`.

    Note:
        :meth:`get_lr` forwards only ``T_mul`` and ``T_mult`` from ``__init__``
        to :meth:`ctxadjust_lr`; the remaining annealing parameters use the
        defaults of :meth:`ctxadjust_lr` (``T_0=15000, eta_min=6e-5,
        eta_max=9e-5, tmctx=0.98, ws=5000``) rather than the constructor
        values. To override those too, call :meth:`ctxadjust_lr` explicitly.
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
            T_mul=1,
            T_mult=2.0,
    ):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.eta_min = eta_min
        self.T_0 = T_0
        self.eta_max = eta_max
        self.T_mul = T_mul
        self.T_mult = T_mult
        super().__init__(optimizer, last_epoch)

    def ctxadjust_lr(self, T_0=15000, eta_min=0.00006, eta_max=0.00009, tmctx=0.98, ws=5000, T_mul=1, T_mult=2.0):
        step_num = self.last_epoch + 1
        if T_mul == 2:
            cycle = int(np.log(step_num * (T_mult - 1) / T_0 + 1) / np.log(T_mult))
            T_cur = step_num - T_0 * (T_mult ** cycle - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** cycle
        else:
            T_cur = (step_num + ws) % T_0
            T_i = T_0
            cycle = (step_num + ws) // T_0
        cur_lr = eta_min * (tmctx ** cycle) + 0.5 * (
                eta_max * (tmctx ** cycle) - eta_min * (tmctx ** cycle)
        ) * (1 + np.cos(np.pi * T_cur / T_i))
        if ws > step_num:
            cur_lr = step_num * (eta_max / ws)
        return cur_lr

    def get_lr(self):
        lrs = []
        for _ in self.base_lrs:
            lrs.append(self.ctxadjust_lr(T_mul=self.T_mul, T_mult=self.T_mult))
        return lrs

    def set_step(self, step: int):
        self.last_epoch = step

