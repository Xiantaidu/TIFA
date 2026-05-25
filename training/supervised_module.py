import pathlib

from torch import nn, Tensor
from torch.utils.data import DataLoader

from lib.config.schema import RootConfig
from training.data import (
    BaseDataset,
    PhonemeTimingDataset,
    TextOnlyDataset,
    DynamicBatchSampler,
    ZippedDataLoader,
)
from training.pl_module_base import BaseLightningModule


class SupervisedModule(BaseLightningModule):

    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        return (
            config.binarizer.phoneme_timing_data_dir_resolved,
            config.binarizer.text_only_data_dir_resolved,
        )

    def build_model(self) -> nn.Module:
        return nn.Linear(1, 1)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("dummy", nn.MSELoss())

    def build_train_dataset(self) -> BaseDataset:
        return PhonemeTimingDataset(
            self.data_dir, "train",
            augmentation_config=self.training_config.augmentation,
        )

    def build_valid_dataset(self) -> BaseDataset:
        if self.use_parallel_dirty_metrics:
            return PhonemeTimingDataset(
                self.data_dir, "valid",
                augmentation_config=self.training_config.augmentation,
                augmentation_deterministic=True,
                augmentation_destructive_only=True,
                augmentation_return_dirty=True,
                ensure_original_tokens=True,
            )
        else:
            return PhonemeTimingDataset(
                self.data_dir, "valid",
                ensure_original_tokens=True,
            )

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.aux_data_dir is not None:
            self.aux_train_dataset = TextOnlyDataset(
                self.aux_data_dir, "aux",
                augmentation_config=self.training_config.augmentation,
            )
            self.aux_train_sampler = None

    def train_dataloader(self):
        main_dl = super().train_dataloader()
        if self.aux_train_dataset is None:
            return ZippedDataLoader(main_dl)

        dl_cfg = self.training_config.dataloader
        multiplier = dl_cfg.aux_multiplier
        self.aux_train_sampler = DynamicBatchSampler(
            self.aux_train_dataset,
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
            self.aux_train_dataset,
            collate_fn=self.aux_train_dataset.collate,
            batch_sampler=self.aux_train_sampler,
            num_workers=dl_cfg.num_workers,
            prefetch_factor=dl_cfg.prefetch_factor if dl_cfg.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=dl_cfg.num_workers > 0,
        )
        return ZippedDataLoader(main_dl, aux_dl, self.aux_train_sampler)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.aux_train_sampler is not None:
            self.aux_train_sampler.set_epoch(self.current_epoch)

    def _is_aux_warmup(self) -> bool:
        """Whether aux dataset exists but hasn't started contributing yet."""
        return (
            self.aux_data_dir is not None
            and self.current_epoch < self.training_config.dataloader.aux_warmup_epochs
        )

    def forward_model(self, sample, infer):
        main_sample = sample["main"]
        aux_sample = sample.get("aux")
        main_sample: dict[str, Tensor]
        aux_sample: dict[str, Tensor] | None
        print(main_sample["spectrogram"].shape)
        print(aux_sample["tokens"].shape if aux_sample is not None else "No aux sample")

        # TODO: real model forward on main_sample and aux_sample.
        # When computing aux losses, zero them during warmup:
        #   if self._is_aux_warmup():
        #       aux_loss = torch.zeros_like(aux_loss)
        raise NotImplementedError("SupervisedModule.forward_model is a stub")

    def plot_validation_results(self, sample, outputs):
        pass
