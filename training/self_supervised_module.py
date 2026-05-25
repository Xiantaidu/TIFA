import pathlib

import torch
from torch import nn

from lib.config.schema import RootConfig
from training.data import (
    BaseDataset,
    TextOnlyDataset,
)
from training.iterative_ranking import RankingModule
from training.pl_module_base import BaseLightningModule


class SelfSupervisedModule(BaseLightningModule, RankingModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_paths: dict[int, dict[int, int]] | None = None

    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        return (
            config.binarizer.text_only_data_dir_resolved,
            None,
        )

    def build_model(self) -> nn.Module:
        return nn.Linear(1, 1)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("dummy", nn.MSELoss())

    def build_train_dataset(self) -> BaseDataset:
        return TextOnlyDataset(
            self.data_dir, "train",
            augmentation_config=self.training_config.augmentation,
        )

    def build_valid_dataset(self) -> BaseDataset:
        return TextOnlyDataset(self.data_dir, "valid")

    def set_best_paths(self, best_paths: dict[int, dict[int, int]] | None) -> None:
        self._best_paths = best_paths

    def training_step(self, sample, batch_index):
        total_loss = super().training_step(sample, batch_index)
        ranking_cfg = self.training_config.iterative_ranking
        if self._best_paths is not None and ranking_cfg.enabled:
            return {
                "loss": total_loss,
                "ranking_results": self._compute_ranking(sample),
            }
        return total_loss

    def forward_model(self, sample, infer):
        if infer:
            return {}

        # TODO: real model forward and loss computation using self._best_paths
        return {"dummy": torch.zeros(1, device=sample["spectrogram"].device,
                                     requires_grad=True)}

    def _compute_ranking(self, sample):
        from lib.path_sampling import sample_paths_uniform
        from training.iterative_ranking import SegmentScores

        widths = sample["widths"]  # [B, S_max] padded with 1
        ranking_cfg = self.training_config.iterative_ranking
        k = ranking_cfg.k
        B = sample["size"]
        S_max = widths.shape[1]

        choices = sample_paths_uniform(widths, k)  # [B, k, S_max]

        # Augment with item index, flatten, dedup at path level
        indices = sample["indices"].view(B, 1, 1).expand(-1, k, -1)  # [B, k, 1]
        augmented = torch.cat([indices, choices], dim=-1)  # [B, k, S_max+1]
        flat = augmented.reshape(B * k, S_max + 1)
        unique_all = torch.unique(flat, dim=0)  # [K', S_max+1]

        item_ids = unique_all[:, 0].long().tolist()  # [K']
        deduped = unique_all[:, 1:].unsqueeze(0)  # [1, K', S_max]

        # Build per-segment score vectors; accumulate over unique paths
        results: list[SegmentScores] = []
        with torch.no_grad():
            for i, item_idx in enumerate(item_ids):
                b = (sample["indices"] == item_idx).nonzero(as_tuple=True)[0].item()
                path = deduped[0, i]  # [S_max]
                for s in range(S_max):
                    w = int(widths[b, s].item())
                    if w <= 1:
                        continue
                    alt_idx = int(path[s].item())
                    score = self.score_subpath(
                        {"spectrogram": sample["spectrogram"][b],
                         "paths": sample["paths"][b],
                         "segments": sample["segments"][b],
                         "N": sample["N"][b].item(),
                         "widths": widths[b]},
                        s, alt_idx,
                    )
                    # Accumulate into existing or new segment vector
                    existing = next(
                        (
                            r for r in results
                            if r.item_idx == item_idx and r.seg_idx == s
                        ),
                        None
                    )
                    if existing is not None:
                        existing.scores[alt_idx] += score
                    else:
                        vec = torch.zeros(w, device=sample["spectrogram"].device)
                        vec[alt_idx] = score
                        results.append(SegmentScores(item_idx, s, vec))
        return results

    def score_subpath(self, item_data: dict,
                      seg_idx: int, alt_idx: int) -> float:
        """Score one subpath for a divergent segment. Higher = better."""
        return 0.0  # TODO: real scoring based on model architecture

    def _extract_subpath_tokens(
            self, paths: torch.Tensor, segments: torch.Tensor,
            seg_idx: int, alt_idx: int, n_grid: int
    ) -> torch.Tensor:
        """Extract the token sequence for one subpath of a specific segment."""
        tokens = []
        for i in range(n_grid):
            s = segments[i].item() - 1
            if s != seg_idx:
                continue
            tok = paths[i, alt_idx].item()
            if tok != 0:
                tokens.append(tok)
        return torch.tensor(tokens, dtype=torch.long)

    def plot_validation_results(self, sample, outputs):
        pass
