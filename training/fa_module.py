import pathlib

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lib.config.schema import RootConfig, LossConfig
from lib.plot import cross_similarity_to_figure, topk_bar_figure
from modules.decoding import decode_alignment
from modules.forced_alignment import ForcedAlignmentModel
from modules.functional import cross_cosine_similarity
from modules.losses.region_loss import FrameAlignmentLoss, SpanContrastiveLoss
from modules.losses.token_loss import TokenAuthenticityLoss
from modules.metrics import (
    BoundaryErrorRate,
    BoundaryMAE,
    PairConjunctionMAE,
    OverlapRatioCollection,
)
from training.data import (
    BaseDataset,
    PhonemeTimingDataset,
    TextOnlyDataset,
    DynamicBatchSampler,
    ZippedDataLoader,
)
from training.pl_module_base import BaseLightningModule, LossValue

# Metric name bases shared between _register_fa_metrics and plot_validation_metrics.
_BER_ONSET = "BER_onset"
_BER_OFFSET = "BER_offset"
_B_MAE_ONSET = "B-MAE_onset"
_B_MAE_OFFSET = "B-MAE_offset"
_OVERLAP = "Overlap"
_CONJ_MAE = "Conj-MAE_"


class ForcedAlignmentModule(BaseLightningModule):

    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        return (
            config.binarizer.phoneme_timing_data_dir_resolved,
            config.binarizer.text_only_data_dir_resolved,
        )

    def build_model(self) -> nn.Module:
        return ForcedAlignmentModel(self.model_config)

    def register_losses_and_metrics(self) -> None:
        loss_cfg: LossConfig = self.training_config.loss

        self.register_loss("frame_alignment", FrameAlignmentLoss(
            temperature=loss_cfg.frame_alignment.temperature,
        ), weight=loss_cfg.frame_alignment.weight)
        self.register_loss("span_contrastive", SpanContrastiveLoss(
            temperature=loss_cfg.span_contrastive.temperature,
            bidirectional=loss_cfg.span_contrastive.bidirectional,
        ), weight=loss_cfg.span_contrastive.weight)

        aug_cfg = self.training_config.augmentation
        if aug_cfg.sequence_edit.enabled:
            self.register_loss(
                "token_authenticity", TokenAuthenticityLoss(), weight=loss_cfg.token_authenticity.weight
            )

        self._register_fa_metrics()
        if self.use_parallel_dirty_metrics:
            self._register_fa_metrics(postfix="_dirty")

    def _register_fa_metrics(self, postfix: str = "") -> None:
        V = self.vocab.vocab_size
        T = self.training_config.validation.metrics_ber_tolerance
        K = self.training_config.validation.metrics_k_values
        KC = self.training_config.validation.metrics_conjunction_k_values

        # Boundary Error Rate -- onset
        self.register_metric(
            f"{_BER_ONSET}{postfix}",
            BoundaryErrorRate(tolerance=T, mode="onset"),
        )
        for k in K:
            self.register_metric(
                f"{_BER_ONSET}@{k}{postfix}",
                BoundaryErrorRate(tolerance=T, mode="onset", vocab_size=V, k=k),
            )

        # Boundary Error Rate -- offset
        self.register_metric(
            f"{_BER_OFFSET}{postfix}",
            BoundaryErrorRate(tolerance=T, mode="offset"),
        )
        for k in K:
            self.register_metric(
                f"{_BER_OFFSET}@{k}{postfix}",
                BoundaryErrorRate(tolerance=T, mode="offset", vocab_size=V, k=k),
            )

        # Boundary MAE -- onset
        self.register_metric(
            f"{_B_MAE_ONSET}{postfix}",
            BoundaryMAE(mode="onset"),
        )
        for k in K:
            self.register_metric(
                f"{_B_MAE_ONSET}@{k}{postfix}",
                BoundaryMAE(mode="onset", vocab_size=V, k=k),
            )

        # Boundary MAE -- offset
        self.register_metric(
            f"{_B_MAE_OFFSET}{postfix}",
            BoundaryMAE(mode="offset"),
        )
        for k in K:
            self.register_metric(
                f"{_B_MAE_OFFSET}@{k}{postfix}",
                BoundaryMAE(mode="offset", vocab_size=V, k=k),
            )

        # Pair Conjunction MAE
        for k in KC:
            self.register_metric(
                f"{_CONJ_MAE}@{k}{postfix}",
                PairConjunctionMAE(vocab_size=V, k=k),
            )

        # Overlap Ratio Collection
        self.register_metric(
            f"{_OVERLAP}{postfix}",
            OverlapRatioCollection(template=f"{_OVERLAP}_{{}}{postfix}"),
        )
        for k in K:
            self.register_metric(
                f"{_OVERLAP}@{k}{postfix}",
                OverlapRatioCollection(
                    template=f"{_OVERLAP}_{{}}@{k}{postfix}", vocab_size=V, k=k,
                ),
            )

    def _update_fa_metrics(
        self, pred_spans: Tensor, target_spans: Tensor, tokens: Tensor, postfix: str = ""
    ) -> None:
        for name, metric in self.metrics.items():
            if name.endswith(postfix):
                metric.update(pred_spans, target_spans, tokens)

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
        sample["indices"] = sample["main"]["indices"]
        super().validation_step(sample, batch_index)

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

        x_features, token_features, token_logits = self.model(spectrogram, tokens, t_mask, n_mask)

        if infer:
            pred_spans = decode_alignment(
                x_features, token_features, T, N,
                temperature=self.training_config.loss.frame_alignment.temperature,
            )
            target_spans = main_sample["spans"]
            self._update_fa_metrics(pred_spans, target_spans, tokens)

            if self.use_parallel_dirty_metrics and "spectrogram_dirty" in main_sample:
                xf_d, tf_d, _ = self.model(
                    main_sample["spectrogram_dirty"], tokens, t_mask, n_mask,
                )
                pred_spans_dirty = decode_alignment(
                    xf_d, tf_d, T, N,
                    temperature=self.training_config.loss.frame_alignment.temperature,
                )
                self._update_fa_metrics(
                    pred_spans_dirty, target_spans, tokens, postfix="_dirty",
                )

            return {
                "x_features": x_features,
                "token_features": token_features,
                "token_logits": token_logits,
            }

        batch_frames = T.sum().item()
        batch_tokens = N.sum().item()
        group_frames = self._group_count(batch_idx, "lengths")
        group_tokens = self._group_count(batch_idx, "tokens")

        frame_alignment_loss = LossValue(
            mean=self.losses["frame_alignment"](x_features, token_features, regions, t_mask, n_mask),
            batch_count=batch_frames, group_count=group_frames,
        )
        span_contrastive_loss = LossValue(
            mean=self.losses["span_contrastive"](x_features, token_features, main_sample["spans"], t_mask, n_mask),
            batch_count=batch_tokens, group_count=group_tokens,
        )
        losses = {
            "frame_alignment": frame_alignment_loss,
            "span_contrastive": span_contrastive_loss,
        }
        if "token_authenticity" in self.losses:
            token_authenticity_loss = LossValue(
                mean=self.losses["token_authenticity"](
                    token_logits, main_sample["authentic"], n_mask,
                ),
                batch_count=batch_tokens, group_count=group_tokens,
            )
            losses["token_authenticity"] = token_authenticity_loss

        return losses

    def plot_validation_results(self, sample, outputs):
        main = sample["main"]
        indices = main["indices"]
        max_plots = self.training_config.validation.max_plots
        stride = max(1, len(self.valid_dataset) // max_plots)

        x_features = outputs["x_features"]
        token_features = outputs["token_features"]
        regions = main["regions"]
        T_all = main["T"]
        N_all = main["N"]

        for i in range(indices.shape[0]):
            data_idx = int(indices[i].item())
            if data_idx % stride != 0:
                continue

            T_i = int(T_all[i].item())
            N_i = int(N_all[i].item())
            if T_i == 0 or N_i == 0:
                continue

            xf = x_features[i, :T_i]
            tf = token_features[i, :N_i]

            token_ids = main["tokens"][i, :N_i].tolist()
            token_labels = [
                self.vocab.decode(int(tid), stringfy=True) or str(tid)
                for tid in token_ids
            ]

            sim = cross_cosine_similarity(xf, tf)  # [T_i, N_i]

            fig = cross_similarity_to_figure(
                sim.float().detach().cpu().numpy().T,  # [N_i, T_i]
                regions=regions[i, :T_i].detach().cpu().numpy(),
                title=f"cross_cosine_similarity (idx={data_idx}, T={T_i}, N={N_i})",
                token_labels=token_labels,
            )
            logger: TensorBoardLogger = self.logger
            logger.experiment.add_figure(
                f"similarity/{data_idx}", fig, global_step=self.global_step,
            )
            plt.close(fig)

    def plot_validation_metrics(self) -> None:
        K = self.training_config.validation.metrics_k_values
        KC = self.training_config.validation.metrics_conjunction_k_values
        postfixes = [""]
        if self.use_parallel_dirty_metrics:
            postfixes.append("_dirty")

        if K:
            k = max(K)
            for pf in postfixes:
                self._plot_boundary_topk(f"{_BER_ONSET}@{k}{pf}")
                self._plot_boundary_topk(f"{_BER_OFFSET}@{k}{pf}")
                self._plot_boundary_topk(f"{_B_MAE_ONSET}@{k}{pf}")
                self._plot_boundary_topk(f"{_B_MAE_OFFSET}@{k}{pf}")
                self._plot_overlap_topk(f"{_OVERLAP}@{k}{pf}")
        if KC:
            k = max(KC)
            for pf in postfixes:
                self._plot_conjunction_topk(f"{_CONJ_MAE}@{k}{pf}")

    def _plot_boundary_topk(self, name: str) -> None:
        data = self.metrics[name].compute_top_k()
        if not data:
            return
        labels = [self.vocab.decode(tid, stringfy=True) or str(tid) for tid in data]
        values = [v.item() for v in data.values()]
        fig = topk_bar_figure(labels, values, name)
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"topk/{name}", fig, global_step=self.global_step)
        plt.close(fig)

    def _plot_conjunction_topk(self, name: str) -> None:
        data = self.metrics[name].compute_top_k()
        if not data:
            return
        labels = [
            f"{self.vocab.decode(i, stringfy=True) or str(i)} -> {self.vocab.decode(j, stringfy=True) or str(j)}"
            for (i, j) in data
        ]
        values = [v.item() for v in data.values()]
        fig = topk_bar_figure(labels, values, name)
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"topk/{name}", fig, global_step=self.global_step)
        plt.close(fig)

    def _plot_overlap_topk(self, name: str) -> None:
        data = self.metrics[name].compute_top_k()
        if not data:
            return
        logger: TensorBoardLogger = self.logger
        for sub_name, sub_data in data.items():
            labels = [self.vocab.decode(tid, stringfy=True) or str(tid) for tid in sub_data]
            values = [v.item() for v in sub_data.values()]
            fig = topk_bar_figure(labels, values, f"{name} {sub_name}", reverse=False)
            logger.experiment.add_figure(
                f"topk/{name}/{sub_name}", fig, global_step=self.global_step,
            )
            plt.close(fig)
