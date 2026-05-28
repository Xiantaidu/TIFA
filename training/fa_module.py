import json
import pathlib

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lib.config.schema import RootConfig, ModelConfig, LossConfig
from modules.forced_alignment import ForcedAlignmentModel
from modules.losses.region_loss import FrameAlignmentLoss, SpanContrastiveLoss
from modules.losses.token_loss import TokenAuthenticityLoss
from training.data import (
    BaseDataset,
    PhonemeTimingDataset,
    TextOnlyDataset,
    DynamicBatchSampler,
    ZippedDataLoader,
)
from training.pl_module_base import BaseLightningModule, LossValue


class ForcedAlignmentModule(BaseLightningModule):

    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        return (
            config.binarizer.phoneme_timing_data_dir_resolved,
            config.binarizer.text_only_data_dir_resolved,
        )

    def build_model(self) -> nn.Module:
        model_config: ModelConfig = self.model_config
        vocab_path = self.data_dir / "vocabulary.json"
        with open(vocab_path, "r", encoding="utf8") as f:
            vocab = json.load(f)
        vocab_size = max(vocab["symbols"].values()) + 1  # +1 for padding index 0
        if vocab_size > model_config.max_vocab_size:
            raise ValueError(
                f"Vocabulary size {vocab_size} exceeds max_vocab_size {model_config.max_vocab_size}"
            )
        return ForcedAlignmentModel(model_config)

    def register_losses_and_metrics(self) -> None:
        loss_cfg: LossConfig = self.training_config.loss

        self.register_loss("frame_alignment", FrameAlignmentLoss(
            temperature=loss_cfg.frame_alignment.temperature,
        ))
        self.register_loss("span_contrastive", SpanContrastiveLoss(
            temperature=loss_cfg.span_contrastive.temperature,
            bidirectional=loss_cfg.span_contrastive.bidirectional,
        ))

        aug_cfg = self.training_config.augmentation
        if aug_cfg.token_perturbation.enabled or aug_cfg.sequence_edit.enabled:
            self.register_loss("token_authenticity", TokenAuthenticityLoss())

    def build_train_dataset(self) -> BaseDataset:
        dl_cfg = self.training_config.dataloader
        return PhonemeTimingDataset(
            self.data_dir, "train",
            augmentation_config=self.training_config.augmentation,
            max_concat_size=dl_cfg.max_concat_size,
            max_concat_frames=dl_cfg.max_concat_frames,
        )

    def build_valid_dataset(self) -> BaseDataset:
        dl_cfg = self.training_config.dataloader
        if self.use_parallel_dirty_metrics:
            aug_cfg = self.training_config.augmentation.keep_destructive()
            return PhonemeTimingDataset(
                self.data_dir, "valid",
                augmentation_config=aug_cfg,
                augmentation_deterministic=True,
                augmentation_return_dirty=True,
                ensure_original_tokens=True,
                max_concat_size=dl_cfg.max_concat_size,
                max_concat_frames=dl_cfg.max_concat_frames,
                concat_deterministic=True,
            )
        else:
            return PhonemeTimingDataset(
                self.data_dir, "valid",
                ensure_original_tokens=True,
                max_concat_size=dl_cfg.max_concat_size,
                max_concat_frames=dl_cfg.max_concat_frames,
                concat_deterministic=True,
            )

    def build_aux_dataset(self) -> BaseDataset | None:
        dl_cfg = self.training_config.dataloader
        return TextOnlyDataset(
            self.aux_data_dir, "aux",
            augmentation_config=self.training_config.augmentation,
            max_concat_size=dl_cfg.max_concat_size,
            max_concat_frames=dl_cfg.max_concat_frames,
        )

    def train_dataloader(self):
        main_dl = super().train_dataloader()
        if self.aux_dataset is None:
            # Always use ZippedDataLoader for consistent sample format in forward_model
            return ZippedDataLoader(main_dl)

        dl_cfg = self.training_config.dataloader
        multiplier = dl_cfg.aux_multiplier
        self.aux_sampler = DynamicBatchSampler(
            self.aux_dataset,
            max_batch_size=int(dl_cfg.max_batch_size * multiplier),
            max_batch_frames=int(dl_cfg.max_batch_frames * multiplier),
            sort_by_len=True,
            frame_count_grid=dl_cfg.frame_count_grid,
            batch_count_multiple_of=self.training_config.trainer.accumulate_grad_batches,
            reassign_batches=True,
            shuffle_batches=False,
            seed=42,
        )
        aux_dl = DataLoader(
            self.aux_dataset,
            collate_fn=self.aux_dataset.collate,
            batch_sampler=self.aux_sampler,
            num_workers=dl_cfg.num_workers,
            prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=dl_cfg.num_workers > 0,
        )
        return ZippedDataLoader(main_dl, aux_dl, self.aux_sampler)

    def val_dataloader(self):
        return ZippedDataLoader(super().val_dataloader())

    def validation_step(self, sample, batch_index):
        super().validation_step(sample["main"], batch_index)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.aux_sampler is not None:
            self.aux_sampler.set_epoch(self.current_epoch)

    def _is_aux_warmup(self) -> bool:
        """Whether aux dataset exists but hasn't started contributing yet."""
        return (
            self.aux_data_dir is not None
            and self.current_epoch < self.training_config.dataloader.aux_warmup_epochs
        )

    def _group_count(self, batch_idx: int | None, key: str) -> int:
        if batch_idx is None:
            return 0  # validation: weight falls back to 1.0 in training_step
        batches = self.get_accumulation_group(batch_idx)
        return sum(
            self.train_dataset.get_metadata(key, idx)
            for batch in batches for idx in batch
        )

    def forward_model(self, sample: dict[str, Tensor], infer: bool, batch_idx=None):
        main_sample = sample["main"]
        # TODO: aux_sample not used yet.
        # When computing aux losses, zero them during warmup:
        #   if self._is_aux_warmup():
        #       aux_loss = torch.zeros_like(aux_loss)

        spectrogram = main_sample["spectrogram"]
        tokens = main_sample["tokens"]
        regions = main_sample["regions"]
        T = main_sample["T"]
        N = main_sample["N"]

        device = spectrogram.device
        B, max_T = spectrogram.shape[:2]
        max_N = tokens.shape[1]

        t_mask = torch.arange(max_T, device=device).unsqueeze(0) < T.unsqueeze(1)
        n_mask = torch.arange(max_N, device=device).unsqueeze(0) < N.unsqueeze(1)

        out_x, out_tok, token_logits = self.model(spectrogram, tokens, t_mask, n_mask)

        if infer:
            return {
                "out_x": out_x,
                "out_tok": out_tok,
                "token_logits": token_logits,
            }

        loss_cfg: LossConfig = self.training_config.loss

        n_frames = (t_mask & (regions > 0)).sum().item()
        n_tokens = n_mask.sum().item()
        group_frames = self._group_count(batch_idx, "regions")
        group_tokens = self._group_count(batch_idx, "tokens")

        losses = {}
        losses["frame_alignment"] = LossValue(
            mean=self.losses["frame_alignment"](out_x, out_tok, regions, t_mask, n_mask)
                 * loss_cfg.frame_alignment.weight,
            batch_count=n_frames, group_count=group_frames,
        )
        losses["span_contrastive"] = LossValue(
            mean=self.losses["span_contrastive"](out_x, out_tok, main_sample["spans"], t_mask, n_mask)
                 * loss_cfg.span_contrastive.weight,
            batch_count=n_tokens, group_count=group_tokens,
        )
        if "token_authenticity" in self.losses:
            losses["token_authenticity"] = LossValue(
                mean=self.losses["token_authenticity"](
                    token_logits, main_sample["authentic"], n_mask,
                ) * loss_cfg.token_authenticity.weight,
                batch_count=n_tokens, group_count=group_tokens,
            )

        return losses

    def plot_validation_results(self, sample, outputs):
        pass
