import math
import pathlib

import librosa
import numpy
import torch

from lib.audio import load_audio
from lib.config.io import load_raw_config
from lib.config.schema import AugmentationConfig, BinarizerFeaturesConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.indexed_dataset import IndexedDataset
from .augmentation import (
    AugmentationContext,
    ComposedAugmentation,
    SpectrogramStretching,
    generate_seed, build_augmentation_chain,
)

__all__ = [
    "collate_nd",
    "BaseDataset",
    "DynamicBatchSampler",
]


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


class BaseDataset(torch.utils.data.Dataset):
    __non_zero_paddings__ = {
        "spectrogram": math.log(1e-5),
        "spectrogram_dirty": math.log(1e-5),
    }

    def __init__(
            self,
            data_dir: pathlib.Path,
            prefix: str,
            augmentation_config: AugmentationConfig = None,
            augmentation_deterministic: bool = False,
            augmentation_destructive_only: bool = False,
            augmentation_return_dirty: bool = False,
    ):
        super().__init__()
        self.info = {
            k: v
            for k, v in numpy.load(data_dir / f"{prefix}.info.npz").items()
        }
        self.data_dir = data_dir
        self.data = IndexedDataset(data_dir, prefix)
        self.epoch = torch.multiprocessing.Value("i", 0)
        self.augmentation_config = augmentation_config
        self.augmentation_deterministic = augmentation_deterministic
        self.augmentation_destructive_only = augmentation_destructive_only
        self.augmentation_return_dirty = augmentation_return_dirty
        self.augmentation_chains: dict[int, ComposedAugmentation] = {}
        self.mel_spectrogram = None
        self._setup()

    def __getitem__(self, index):
        sample = self.data[index]
        waveform = self._load_waveform(index)
        spectrogram_clean = None

        augmentation = {}
        if self.augmentation_config is not None:
            chain = self.augmentation_chains[index]

            wf_tensor = torch.from_numpy(waveform).unsqueeze(0)
            if self.augmentation_return_dirty:
                spectrogram_clean = self.mel_spectrogram(wf_tensor).squeeze(0).T

            ctx = AugmentationContext(waveform=waveform, sr=self.sample_rate)
            chain.apply(ctx)
            spectrogram = ctx.spectrogram
            augmentation = chain.args_dict()
        else:
            wf_tensor = torch.from_numpy(waveform).unsqueeze(0)
            spectrogram = self.mel_spectrogram(wf_tensor).squeeze(0).T

        spectrogram = torch.clamp(spectrogram, min=math.log(1e-5))

        if self.augmentation_return_dirty:
            sample["spectrogram"] = torch.clamp(spectrogram_clean, min=math.log(1e-5))
            sample["spectrogram_dirty"] = spectrogram
        else:
            sample["spectrogram"] = spectrogram

        return {
            "_idx": index,
            "_name": self.info["item_paths"][index],
            "_augmentation": augmentation,
            **sample,
        }

    def __len__(self):
        return self.info["lengths"].shape[0]

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        if self.augmentation_config is not None and not self.augmentation_deterministic:
            self._build_chains(numpy.random.default_rng())

    def num_frames(self, index: int) -> int:
        base_len = int(self.info["lengths"][index])
        chain = self.augmentation_chains.get(index)
        if chain is not None:
            for t in chain.transforms:
                if isinstance(t, SpectrogramStretching) and t.speed is not None:
                    return max(1, int(base_len / t.speed))
        return base_len

    def _setup(self):
        feature_raw = load_raw_config(self.data_dir / "feature.yaml")
        feature_cfg = BinarizerFeaturesConfig.model_validate(feature_raw)
        self.sample_rate = feature_cfg.audio_sample_rate
        self.mel_spectrogram = StretchableMelSpectrogram(
            sample_rate=feature_cfg.audio_sample_rate,
            n_mels=feature_cfg.spectrogram.num_bins,
            n_fft=feature_cfg.fft_size,
            win_length=feature_cfg.win_size,
            hop_length=feature_cfg.hop_size,
            fmin=feature_cfg.spectrogram.fmin,
            fmax=feature_cfg.spectrogram.fmax,
            clip_val=1e-9,
        ).eval()

        if self.augmentation_config is None:
            return

        if self.augmentation_deterministic:
            seed = generate_seed(sorted(self.info.keys()))
            self._build_chains(numpy.random.default_rng(seed))

    def _build_chains(self, generator: numpy.random.Generator):
        self.augmentation_chains.clear()
        for index in range(len(self)):
            self.augmentation_chains[index] = build_augmentation_chain(
                self.augmentation_config, generator=generator,
                mel_spectrogram=self.mel_spectrogram,
                destructive_only=self.augmentation_destructive_only,
            )

    def _load_waveform(self, index: int) -> numpy.ndarray:
        waveform_fn = self.data_dir / str(self.info["item_paths"][index])
        waveform, sr = load_audio(waveform_fn)
        if sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
        return waveform

    @classmethod
    def collate(cls, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        for s in samples:
            s.pop("_augmentation", None)
        batch = {
            "size": len(samples),
            "indices": torch.LongTensor([s.pop("_idx") for s in samples]),
            "names": [s.pop("_name") for s in samples],
        }
        if len(samples) == 0:
            return batch
        for key, value in samples[0].items():
            if value.ndim == 0:
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                pad_value = cls.__non_zero_paddings__.get(key, 0)
                batch[key] = collate_nd([s[key] for s in samples], pad_value=pad_value)
        return batch


class DynamicBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
            self,
            dataset: BaseDataset,
            max_batch_size: int,
            max_batch_frames: int,
            sort_by_len: bool = True,
            frame_count_grid: int = 1,
            batch_count_multiple_of: int = 1,
            reassign_batches: bool = True,
            shuffle_batches: bool = True,
            seed: int = 0,
    ):
        if torch.distributed.is_initialized():
            num_replicas = None
            rank = None
        else:
            num_replicas = 1
            rank = 0
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_batch_frames = max_batch_frames
        self.sort_by_len = sort_by_len
        self.frame_count_grid = frame_count_grid
        self.batch_count_multiple_of = batch_count_multiple_of
        self.reassign_batches = reassign_batches
        self.shuffle_batches = shuffle_batches
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.batches: list[list[int]] = None
        self.formed = None

    def __iter__(self):
        self.form_batches()
        return iter(self.batches)

    def __len__(self):
        self.form_batches()
        return len(self.batches)

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        self.generator = torch.Generator().manual_seed(self.seed + epoch)

    def permutation(self, n: int) -> list[int]:
        perm = torch.randperm(n, generator=self.generator).tolist()
        return perm

    def form_batches(self):
        if self.formed == self.epoch + self.seed:
            return

        self.dataset.set_epoch(self.epoch)
        lengths = [self.dataset.num_frames(i) for i in range(len(self.dataset))]
        if self.sort_by_len:
            sorted_indices = sorted(
                self.permutation(len(lengths)),
                key=lambda x: lengths[x] // self.frame_count_grid, reverse=True
            )
        else:
            sorted_indices = list(range(len(lengths)))

        def batch_full(batch_: list[int], new_index_: int):
            if len(batch_) >= self.max_batch_size:
                return True
            max_len = max(lengths[new_index_], max((lengths[i] for i in batch_), default=0))
            if max_len * (len(batch_) + 1) > self.max_batch_frames:
                return True
            return False

        batches: list[list[int]] = []

        current_batch = []
        for idx in sorted_indices:
            sample_length = lengths[idx]
            if sample_length > self.max_batch_frames:
                raise ValueError(
                    f"Sample length {sample_length} exceeds max batch frames {self.max_batch_frames}."
                )
            if batch_full(current_batch, idx):
                batches.append(current_batch)
                current_batch = []
            current_batch.append(idx)
        if current_batch:
            batches.append(current_batch)

        multiple_of = self.num_replicas * self.batch_count_multiple_of
        remainder = (multiple_of - (len(batches) % multiple_of)) % multiple_of
        if self.reassign_batches:
            new_batch = []
            while remainder > 0:
                num_batches = len(batches)
                perm = self.permutation(num_batches)
                batches = [batches[i] for i in perm]
                modified = False
                idx = 0
                while remainder > 0 and idx < num_batches:
                    batch = batches[idx]
                    if len(batch) > 1:
                        item = batch[-1]
                        if batch_full(new_batch, item):
                            batches.append(new_batch)
                            new_batch = []
                            modified = True
                            remainder -= 1
                            if remainder == 0:
                                break
                        batch.pop()
                        new_batch.append(item)
                    idx += 1
                if not modified:
                    if len(new_batch) > 0:
                        batches.append(new_batch)
                        new_batch = []
                        remainder -= 1
                    if remainder > 0:
                        raise RuntimeError(
                            f"Unable to reassign batches to meet the required multiple count of {multiple_of}."
                        )
        else:
            batches += [[]] * remainder

        if self.shuffle_batches:
            perm = self.permutation(len(batches))
            batches = [batches[i] for i in perm]
        elif self.sort_by_len:
            batches = sorted(
                batches,
                key=lambda b: len(b) * max(lengths[i] for i in b),
                reverse=True
            )

        batches = [b for i, b in enumerate(batches) if i % self.num_replicas == self.rank]

        self.batches = batches
        self.formed = self.epoch + self.seed
