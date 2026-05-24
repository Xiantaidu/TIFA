import random
from typing import Any

import lightning.pytorch
import torch
import torch.distributed
from torch import Tensor


def rank_rewards(k: int) -> Tensor:
    """Rank-based rewards with step 2 centered on zero. k=4 → [-3,-1,1,3]."""
    return torch.arange(-(k - 1), k, 2, dtype=torch.float32)


def sample_paths(widths: list[int], k: int) -> list[list[int]]:
    """Sample up to *k* distinct paths by uniformly picking an alternative per segment."""
    paths_set: set[tuple[int, ...]] = set()
    max_attempts = k * 10
    for _ in range(max_attempts):
        path = tuple(
            random.randrange(w) if w > 1 else 0
            for w in widths
        )
        paths_set.add(path)
        if len(paths_set) >= k:
            break
    return [list(p) for p in paths_set]


class PathRanker:
    """Per-subpath accumulated scores with sparse auto-creation.

    Scores are stored as ``{item_idx: {seg_idx: Tensor([alt0_score, ...])}}``.
    Items and segments are created on first access via ``update_scores``.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self._scores: dict[int, dict[int, Tensor]] = {}

    def __len__(self) -> int:
        return len(self._scores)

    def get_best_path(self, item_idx: int) -> list[int]:
        """Highest-scoring alternative index per segment. Empty list if unknown."""
        segs = self._scores.get(item_idx)
        if not segs:
            return []
        return [
            int(segs[p].argmax().item()) if segs[p].numel() > 1 else 0
            for p in sorted(segs)
        ]

    def update_scores(self, item_idx: int, paths: list[list[int]], rewards: Tensor):
        """Distribute per-path rewards equally across all segments, clamped ≥ 0."""
        num_seg = max((len(p) for p in paths), default=0)
        if num_seg == 0:
            return
        for path, reward in zip(paths, rewards):
            r = reward.item()
            if r == 0:
                continue
            delta = r / num_seg
            self._ensure_item(item_idx, num_seg, path)
            segs = self._scores[item_idx]
            for p in range(num_seg):
                alt = path[p]
                segs[p][alt] = max(segs[p][alt].item() + delta, 0.0)

    def _ensure_item(self, item_idx: int, num_seg: int, path: list[int]):
        if item_idx not in self._scores:
            self._scores[item_idx] = {}
        segs = self._scores[item_idx]
        for p in range(num_seg):
            if p not in segs:
                segs[p] = torch.zeros(path[p] + 1)
            elif segs[p].numel() <= path[p]:
                old = segs[p]
                new = torch.zeros(path[p] + 1)
                new[:len(old)] = old
                segs[p] = new

    def decay_scores(self):
        for segs in self._scores.values():
            for seg_scores in segs.values():
                seg_scores.mul_(self.gamma)

    def sync_batch_scores(self, device: torch.device, item_indices: list[int]):
        """All-gather batch item indices across DDP ranks, then all-reduce SUM
        only the scores for items touched by any rank."""
        if not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            return

        all_lists: list[list[int]] = [None] * world_size
        torch.distributed.all_gather_object(all_lists, item_indices)
        union: set[int] = set()
        for lst in all_lists:
            union.update(lst)

        tensors: list[Tensor] = []
        for idx in sorted(union):
            segs = self._scores.get(idx)
            if segs is None:
                continue
            for p in sorted(segs):
                tensors.append(segs[p])
        if not tensors:
            return
        flat = torch.cat([t.flatten() for t in tensors]).to(device)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat = flat.cpu()
        offset = 0
        for idx in sorted(union):
            segs = self._scores.get(idx)
            if segs is None:
                continue
            for p in sorted(segs):
                n = segs[p].numel()
                segs[p].copy_(flat[offset:offset + n])
                offset += n

    def state_dict(self) -> dict[str, Tensor]:
        """``{"itemIdx,segIdx": per_alternative_scores}``."""
        sd: dict[str, Tensor] = {}
        for i, segs in self._scores.items():
            for p, seg_scores in segs.items():
                sd[f"{i},{p}"] = seg_scores.clone()
        return sd

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self._scores.clear()
        for key, tensor in state_dict.items():
            i_str, p_str = key.split(",")
            i = int(i_str)
            p = int(p_str)
            self._scores.setdefault(i, {})[p] = tensor


class IterativeRanking(lightning.pytorch.Callback):
    """Per-step path ranking with epoch-boundary decay and DDP sync.

    Reads ``pl_module._ranking_results`` each batch to update subpath scores,
    pushes ``pl_module._best_paths`` so the module knows which paths to use.
    """

    def __init__(self, k: int, gamma: float, update_every_n_epochs: int):
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.update_every_n_epochs = update_every_n_epochs
        self.ranker: PathRanker | None = None
        self._best_paths: dict[int, list[int]] | None = None
        self._pending_ranker_state: dict[str, Tensor] | None = None

    def on_train_start(self, trainer, pl_module):
        if pl_module.aux_train_dataset is None:
            return
        self.ranker = PathRanker(self.gamma)
        if self._pending_ranker_state is not None:
            self.ranker.load_state_dict(self._pending_ranker_state)
            self._pending_ranker_state = None
        self._best_paths = {}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module._best_paths is None:
            pl_module._best_paths = self._best_paths

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ranker is None:
            return
        results = pl_module._ranking_results
        batch_indices: list[int] = []
        if results is not None:
            with torch.no_grad():
                for r in results:
                    item_idx = r["item_idx"]
                    paths = r["paths"]
                    scores = r["scores"]
                    k = len(paths)
                    _, sorted_idx = torch.sort(scores, descending=True)
                    rewards = rank_rewards(k)
                    mapped_rewards = torch.zeros(k)
                    mapped_rewards[sorted_idx] = rewards
                    self.ranker.update_scores(item_idx, paths, mapped_rewards)
                    batch_indices.append(item_idx)
        self.ranker.sync_batch_scores(pl_module.device, batch_indices)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.ranker is None:
            return
        if trainer.current_epoch > 0 and trainer.current_epoch % self.update_every_n_epochs == 0:
            self.ranker.decay_scores()
            self._best_paths = {
                i: self.ranker.get_best_path(i)
                for i in self.ranker._scores
            }
            pl_module._best_paths = self._best_paths

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        if self.ranker is not None:
            sd["iterative_ranker"] = self.ranker.state_dict()
        if self._best_paths is not None:
            sd["iterative_best_paths"] = {
                str(i): p for i, p in self._best_paths.items()
            }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ranker state, deferring if ``on_train_start`` hasn't run yet."""
        super().load_state_dict(state_dict)
        pending = state_dict.get("iterative_ranker")
        if pending is None:
            return
        if self.ranker is not None:
            self.ranker.load_state_dict(pending)
            saved_paths = state_dict.get("iterative_best_paths")
            if saved_paths is not None:
                self._best_paths = {int(i): p for i, p in saved_paths.items()}
            else:
                self._best_paths = {
                    i: self.ranker.get_best_path(i)
                    for i in self.ranker._scores
                }
        else:
            self._pending_ranker_state = pending
