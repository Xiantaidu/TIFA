import pathlib

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, Tensor

from lib.config.schema import RootConfig, LossConfig
from lib.path_traversal import sample_paths_uniform
from lib.plot import alignment_to_figure, emission_to_figure, reconstruction_to_figure
from lib.vocabulary import SPACE_TOKEN
from modules.commons.common_layers import TemporalMask
from modules.decoding import decode_alignment_spaced
from modules.forced_alignment import ForcedAlignmentSSLModel
from modules.functional import interleave_spaces
from modules.losses import HMMForwardLossWithEmissions, SpectrogramReconstructionLoss
from training.data import (
    BaseDataset,
    TextOnlyDataset,
)
from training.iterative_ranking import RankingModule, SegmentRewards, rank_rewards
from training.pl_module_base import BaseLightningModule, LossValue

# Loss names shared between register_losses_and_metrics and forward_model.
_HMM_FORWARD = "hmm_forward_loss"
_RECONSTRUCTION = "reconstruction_loss"


class ForcedAlignmentSSLModule(BaseLightningModule, RankingModule):

    def __init__(self, *args, **kwargs):
        self.spec_mask = None
        super().__init__(*args, **kwargs)
        self._best_paths: dict[int, dict[int, int]] | None = None

    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        return (
            config.binarizer.text_only_data_dir_resolved,
            None,
        )

    def build_model(self) -> nn.Module:
        return ForcedAlignmentSSLModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        loss_cfg: LossConfig = self.training_config.loss

        self.register_loss(
            _HMM_FORWARD, HMMForwardLossWithEmissions(
                mode=loss_cfg.hmm_forward.mode,
            ),
            weight=loss_cfg.hmm_forward.weight,
        )
        self.register_loss(
            _RECONSTRUCTION, SpectrogramReconstructionLoss(
                loss_type=loss_cfg.reconstruction.loss_type,
            ),
            weight=loss_cfg.reconstruction.weight,
        )

    def post_init(self) -> None:
        self.spec_mask = TemporalMask(
            channels=self.model_config.in_dim,
            seed=42
        )

    def build_train_dataset(self) -> BaseDataset:
        dl_cfg = self.training_config.dataloader
        return TextOnlyDataset(
            self.data_dir, "train",
            augmentation_config=self.training_config.augmentation,
            max_concat_size=dl_cfg.max_concat_size,
            max_concat_frames=dl_cfg.max_concat_frames,
        )

    def build_valid_dataset(self) -> BaseDataset:
        dl_cfg = self.training_config.dataloader
        return TextOnlyDataset(
            self.data_dir, "valid",
            max_concat_size=dl_cfg.max_concat_size,
            max_concat_frames=dl_cfg.max_concat_frames,
            concat_deterministic=True,
        )
        if self.use_parallel_dirty_metrics:
            aug_cfg = self.training_config.augmentation.keep_destructive()
            return TextOnlyDataset(
                self.data_dir, "valid",
                augmentation_config=aug_cfg,
                augmentation_deterministic=True,
                augmentation_return_dirty=True,
                max_concat_size=dl_cfg.max_concat_size,
                max_concat_frames=dl_cfg.max_concat_frames,
                concat_deterministic=True,
            )
        else:
            return TextOnlyDataset(
                self.data_dir, "valid",
                max_concat_size=dl_cfg.max_concat_size,
                max_concat_frames=dl_cfg.max_concat_frames,
                concat_deterministic=True,
            )

    def _group_count(self, batch_idx: int | None, key: str) -> int:
        if batch_idx is None:
            return 0
        batches = self.get_accumulation_group(batch_idx)
        return sum(
            self.train_dataset.get_metadata(key, idx)
            for batch in batches for idx in batch
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.spec_mask.reset_random_generator()

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

    def forward_model(self, sample: dict[str, Tensor], infer: bool, batch_idx=None):
        spectrogram = sample["spectrogram"]  # [B, T, n_mels]
        paths = sample["paths"]  # [B, N, W]
        T = sample["T"]  # [B] int64
        N = sample["N"]  # [B] int64
        f0 = sample["f0"]  # [B, T]

        device = spectrogram.device
        _, max_T, _ = spectrogram.shape

        # Select first path alternative, interleave space tokens
        tokens_raw = paths[:, :, 0]  # [B, N]
        tokens, n_mask = interleave_spaces(tokens_raw, space=SPACE_TOKEN, lengths=N)
        # tokens: [B, 2*N+1], n_mask: [B, 2*N+1] bool

        # Frame mask
        t_mask = torch.arange(max_T, device=device).unsqueeze(0) < T.unsqueeze(1)  # [B, T]

        token_lens = 2 * N + 1

        if infer:
            _, _, attn_logits, _ = self.model(
                spectrogram, tokens, t_mask, n_mask, reconstruct=False,
            )
            # Mean all heads across all CA layers -> [B, T, N_interleaved]
            all_flat = []
            for a in attn_logits:
                mid = a.dim() - 3
                if mid > 0:
                    all_flat.append(a.flatten(1, mid))
                else:
                    all_flat.append(a.unsqueeze(1))
            emission = torch.cat(all_flat, dim=1).mean(dim=1)  # [B, T, N_interleaved]
            attn_weights = emission.softmax(dim=-1)

            # Masked forward for reconstruction visualization
            masked_spec = self.spec_mask(spectrogram, mask=t_mask)
            _, _, _, recon_spec = self.model(
                masked_spec, tokens, t_mask, n_mask, f0=f0, reconstruct=True,
            )

            pred_spans = decode_alignment_spaced(emission, T, token_lens)
            return {
                "spans": pred_spans,
                "tokens": tokens,
                "attn_weights": attn_weights,
                "masked_spec": masked_spec,
                "recon_spec": recon_spec,
            }

        # Apply temporal masking (self-supervised corruption)
        masked_spec = self.spec_mask(spectrogram, mask=t_mask)

        # Model forward
        _, _, attn_logits, x_recon = self.model(
            masked_spec, tokens, t_mask, n_mask, f0=f0, reconstruct=True,
        )

        # Gradient accumulation counts (frame-based for both losses)
        batch_count = int(T.sum().item())
        group_count = self._group_count(batch_idx, "lengths")

        # HMM loss from cross-attention attn_logits
        hmm_loss_val = self.losses[_HMM_FORWARD](
            attn_logits, frame_lens=T, token_lens=token_lens,
        )
        hmm_loss = LossValue(
            mean=hmm_loss_val, batch_count=batch_count, group_count=group_count,
        )

        # Reconstruction loss against clean spectrogram
        recon_loss_val = self.losses[_RECONSTRUCTION](
            x_recon=x_recon, target=spectrogram, t_mask=t_mask,
        )
        recon_loss = LossValue(
            mean=recon_loss_val, batch_count=batch_count, group_count=group_count,
        )

        return {
            _HMM_FORWARD: hmm_loss,
            _RECONSTRUCTION: recon_loss,
        }

    def _compute_ranking(self, sample):
        widths = sample["widths"]  # [B, S_max] padded with 1
        rank_size = self.training_config.iterative_ranking.rank_size
        B = sample["size"]
        S_max = widths.shape[1]
        device = sample["spectrogram"].device

        choices = sample_paths_uniform(widths, rank_size)  # [B, rank_size, S_max]

        results: list[SegmentRewards] = []
        with torch.no_grad():
            for b in range(B):
                item_idx = int(sample["indices"][b].item())
                item_data = {
                    "spectrogram": sample["spectrogram"][b],
                    "paths": sample["paths"][b],
                    "segments": sample["segments"][b],
                    "N": sample["N"][b].item(),
                    "widths": widths[b],
                }
                for s in range(S_max):
                    w = int(widths[b, s].item())
                    if w <= 1:
                        continue
                    # rank_size alt choices for this segment, one per sampled path
                    alts = choices[b, :, s]  # [rank_size]
                    # Score each alt, then rank best->worst
                    scores = torch.zeros(rank_size, device=device, dtype=torch.long)
                    for j in range(rank_size):
                        scores[j] = self.score_subpath(
                            item_data, s, int(alts[j].item()),
                        )
                    _, rank_order = scores.sort(descending=True)
                    rewards = rank_rewards(rank_size).to(device)
                    # Accumulate rank-based rewards into segment vector
                    vec = torch.zeros(w, device=device, dtype=torch.long)
                    for j in range(rank_size):
                        alt_idx = int(alts[rank_order[j]].item())
                        vec[alt_idx] += rewards[j].item()
                    results.append(SegmentRewards(item_idx, s, vec))
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
        indices = sample["indices"]
        spectrograms = sample["spectrogram"]
        T_all = sample["T"]
        N_all = sample["N"]
        tokens_all = outputs["tokens"]  # [B, 2*N+1] interleaved
        pred_spans_all = outputs["spans"]          # [B, 2*N+1, 2]
        attn_all = outputs["attn_weights"]         # [B, T, N_interleaved]

        for i in range(indices.shape[0]):
            data_idx = int(indices[i].item())
            if data_idx not in self.plot_indices:
                continue
            T_i = int(T_all[i].item())
            N_i = int(N_all[i].item())
            if T_i == 0 or N_i == 0:
                continue

            N_interleaved = 2 * N_i + 1

            # All token labels (interleaved: SP for space, decode for real)
            token_labels = []
            for j in range(N_interleaved):
                if j % 2 == 0:
                    token_labels.append("SP")
                else:
                    tid = int(tokens_all[i, j].item())
                    label = self.vocab.decode(tid, stringfy=True) or str(tid)
                    token_labels.append(label)

            # Real token labels (odd indices only)
            real_labels = token_labels[1::2]

            item_path = self.valid_dataset.get_metadata("item_paths", data_idx)
            logger: TensorBoardLogger = self.logger

            # Emission heatmap
            attn = attn_all[i, :T_i, :N_interleaved].float().detach().cpu().numpy()
            fig_em = emission_to_figure(
                attn.T,  # [N_interleaved, T_i]
                title=item_path,
                token_labels=token_labels,
                vmin=0.0,
                label="attention",
            )
            logger.experiment.add_figure(
                f"emission/{data_idx}", fig_em, global_step=self.global_step,
            )
            plt.close(fig_em)

            # Alignment plot: real tokens only
            spec = spectrograms[i, :T_i].detach().cpu().numpy()
            ps = pred_spans_all[i, 1::2].detach().cpu().numpy()  # [N_i, 2] odd indices
            fig_align = alignment_to_figure(
                spec, token_labels=real_labels, pred_spans=ps,
                title=item_path,
            )
            logger.experiment.add_figure(
                f"alignment/{data_idx}", fig_align, global_step=self.global_step,
            )
            plt.close(fig_align)

            # Reconstruction plot: original, masked, reconstructed, |diff|
            spec_masked = outputs["masked_spec"][i, :T_i].detach().cpu().numpy()
            spec_recon = outputs["recon_spec"][i, :T_i].detach().cpu().numpy()
            fig_recon = reconstruction_to_figure(
                spec, spec_masked, spec_recon,
                title=item_path,
            )
            logger.experiment.add_figure(
                f"reconstruction/{data_idx}", fig_recon, global_step=self.global_step,
            )
            plt.close(fig_recon)
