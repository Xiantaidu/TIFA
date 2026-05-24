from typing import Any

import lightning.pytorch
import torch
import torch.distributed
from torch import Tensor


def rank_rewards(k: int) -> Tensor:
    """Rank-based rewards with step 2 centered on zero, best first. k=4 → [3,1,-1,-3]."""
    return torch.arange(k - 1, -k, -2, dtype=torch.float32)


class PathRanker:
    """Per-subpath accumulated scores with sparse auto-creation.

    Stores ``{item_idx: {seg_idx: Tensor([alt0_score, ...])}}``.
    Receives flat ``(item_idx, seg_idx, alt_idx, score)`` tuples,
    groups by ``(item_idx, seg_idx)``, ranks within each group,
    and assigns ``rank_rewards(group_size)`` to each alt.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self._scores: dict[int, dict[int, Tensor]] = {}

    def __len__(self) -> int:
        return len(self._scores)

    def get_best_path(self, item_idx: int) -> dict[int, int]:
        """``{seg_idx: best_alt_idx}`` for known divergent segments. Empty if unknown."""
        segs = self._scores.get(item_idx)
        if not segs:
            return {}
        return {p: int(s.argmax().item()) for p, s in segs.items()}

    def update_scores(self, results: list[tuple[int, int, int, float]]):
        """Group by (item_idx, seg_idx), rank by score, assign rewards."""
        groups: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for item_idx, seg_idx, alt_idx, score in results:
            groups.setdefault((item_idx, seg_idx), []).append((alt_idx, score))

        for (item_idx, seg_idx), entries in groups.items():
            scores = torch.tensor([s for _, s in entries], dtype=torch.float32)
            _, sorted_idx = torch.sort(scores, descending=True)
            rewards = rank_rewards(len(entries))
            mapped = torch.zeros(len(entries))
            mapped[sorted_idx] = rewards

            self._ensure_segment(item_idx, seg_idx, len(entries))
            seg_scores = self._scores[item_idx][seg_idx]
            for (alt_idx, _), r in zip(entries, mapped):
                r_val = r.item()
                if r_val == 0:
                    continue
                seg_scores[alt_idx] = max(seg_scores[alt_idx].item() + r_val, 0.0)

    def _ensure_segment(self, item_idx: int, seg_idx: int, width: int):
        if item_idx not in self._scores:
            self._scores[item_idx] = {}
        segs = self._scores[item_idx]
        if seg_idx not in segs:
            segs[seg_idx] = torch.zeros(width)
        elif segs[seg_idx].numel() < width:
            old = segs[seg_idx]
            new = torch.zeros(width)
            new[:len(old)] = old
            segs[seg_idx] = new

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

    def __init__(self, gamma: float, update_every_n_epochs: int):
        super().__init__()
        self.gamma = gamma
        self.update_every_n_epochs = update_every_n_epochs
        self.ranker: PathRanker | None = None
        self._best_paths: dict[int, dict[int, int]] | None = None
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
        results: list[tuple[int, int, int, float]] | None = pl_module._ranking_results
        batch_indices: list[int] = []
        if results is not None:
            with torch.no_grad():
                self.ranker.update_scores(results)
                for item_idx, _, _, _ in results:
                    if item_idx not in batch_indices:
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
                str(i): {str(p): a for p, a in segs.items()}
                for i, segs in self._best_paths.items()
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
                self._best_paths = {
                    int(i): {int(p): a for p, a in segs.items()}
                    for i, segs in saved_paths.items()
                }
            else:
                self._best_paths = {
                    i: self.ranker.get_best_path(i)
                    for i in self.ranker._scores
                }
        else:
            self._pending_ranker_state = pending
