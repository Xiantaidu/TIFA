from typing import Union

import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler


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

    The learning rate has two phases (see :meth:`_compute_lr` for the actual
    computation):

    1. Warmup phase (``step < warmup_steps``):
        Linear ramp from 0 up to ``max_lr``::

            lr = step * (max_lr / warmup_steps)

    2. Cosine annealing with periodic warm restarts (``step >= warmup_steps``):
        Let ``t = step - warmup_steps`` be the number of steps elapsed since
        warmup finished. Within a cycle the learning rate follows a cosine
        curve smoothly decreasing from the upper bound to the lower bound, then
        warm-restarts back to the upper bound at the end of the cycle. The
        bounds of the ``k``-th cycle are both scaled by ``peak_decay ** k`` so
        that the peak of each cycle decreases over time. Two cycle-length
        schemes are available, selected by ``cycle_mult``:

        * ``cycle_mult == 1`` (default): fixed-length cycles, every cycle lasts
          ``cycle_steps`` steps::

            k     = t // cycle_steps
            T_cur = t %  cycle_steps
            T_i   = cycle_steps

        * ``cycle_mult > 1``: geometrically growing cycles, the ``k``-th cycle
          lasts ``cycle_steps * cycle_mult ** k`` steps::

            k     = floor(log(t * (cycle_mult - 1) / cycle_steps + 1) / log(cycle_mult))
            T_cur = t - cycle_steps * (cycle_mult ** k - 1) / (cycle_mult - 1)
            T_i   = cycle_steps * cycle_mult ** k

        In both cases the learning rate is::

            lo = (min_lr * peak_decay ** k) if decay_floor else min_lr
            hi = max_lr * peak_decay ** k
            lr = lo + 0.5 * (hi - lo) * (1 + cos(pi * T_cur / T_i))

        Because ``t`` is measured from the end of warmup, the first cosine cycle
        starts at ``T_cur == 0`` exactly when warmup finishes, so the schedule
        is continuous (warmup ends at ``max_lr`` and the cosine starts there
        too -- no gap/jump).

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear-warmup steps before cosine annealing.
        max_lr: Peak learning rate (top of the first cosine cycle).
        min_lr: Trough learning rate (bottom of the first cosine cycle).
        cycle_steps: Number of steps in the first cosine annealing cycle.
        peak_decay: Multiplier applied to the bounds at every restart, e.g.
            ``0.98`` lowers them by 2% per cycle. Use ``1.0`` to disable decay
            (classic SGDR).
        decay_floor: If ``True`` (default) both the peak and the trough decay
            by ``peak_decay`` each restart. If ``False`` only the peak decays
            and the trough stays fixed at ``min_lr``.
        cycle_mult: Cycle-length scheme. ``1.0`` for fixed-length cycles;
            ``> 1.0`` for geometrically growing cycles (e.g. ``2.0`` doubles the
            cycle length after each restart).
        last_epoch: Index of the previous step; ``-1`` starts from scratch.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 2500,
            max_lr: float = 4e-4,
            min_lr: float = 5e-5,
            cycle_steps: Union[int, float] = 15000,
            peak_decay: float = 0.98,
            cycle_mult: float = 1.0,
            decay_floor: bool = True,
            last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_steps = cycle_steps
        self.peak_decay = peak_decay
        self.cycle_mult = cycle_mult
        self.decay_floor = decay_floor
        super().__init__(optimizer, last_epoch)

    def _compute_lr(self, step_num: int) -> float:
        # Phase 1: linear warmup.
        if step_num < self.warmup_steps:
            return step_num * (self.max_lr / self.warmup_steps)

        # Phase 2: cosine annealing with warm restarts, counted from the end
        # of warmup so that the first cycle starts at its peak (no gap).
        t = step_num - self.warmup_steps
        if self.cycle_mult == 1.0:
            cycle = t // self.cycle_steps
            T_cur = t % self.cycle_steps
            T_i = self.cycle_steps
        else:
            cycle = int(np.log(t * (self.cycle_mult - 1) / self.cycle_steps + 1) / np.log(self.cycle_mult))
            T_cur = t - self.cycle_steps * (self.cycle_mult ** cycle - 1) / (self.cycle_mult - 1)
            T_i = self.cycle_steps * self.cycle_mult ** cycle

        decay = self.peak_decay ** cycle
        hi = self.max_lr * decay
        lo = self.min_lr * decay if self.decay_floor else self.min_lr
        return lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * T_cur / T_i))

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = self._compute_lr(step_num)
        return [lr for _ in self.base_lrs]

    def set_step(self, step: int):
        self.last_epoch = step

