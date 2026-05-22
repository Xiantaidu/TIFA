import math

import torch
from torch import nn

from training.data import BaseDataset
from training.pl_module_base import BaseLightningModule


class SupervisedDataset(BaseDataset):
    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        # sample keys: spectrogram [T_spec, F], tokens [N], spans [N,2], regions [T]

        T_spec = sample["spectrogram"].shape[0]
        T = sample["regions"].shape[0]

        if T_spec > T:
            sample["spectrogram"] = sample["spectrogram"][:T]
        elif T_spec < T:
            sample["spectrogram"] = torch.nn.functional.pad(
                sample["spectrogram"], (0, 0, 0, T - T_spec),
                value=math.log(1e-5),
            )

        sample["T"] = torch.tensor(max(T_spec, T), dtype=torch.long)
        sample["N"] = torch.tensor(sample["tokens"].shape[0], dtype=torch.long)
        return sample


class SupervisedModule(BaseLightningModule):
    __dataset__ = SupervisedDataset

    def build_model(self) -> nn.Module:
        return nn.Linear(1, 1)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("dummy", nn.MSELoss())

    def forward_model(self, sample, infer):
        raise NotImplementedError("SupervisedModule.forward_model is a stub")

    def plot_validation_results(self, sample, outputs):
        pass
