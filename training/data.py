import abc
import math
import pathlib
import random

import librosa
import numpy
import torch

from lib.audio import load_audio
from lib.config.io import load_raw_config
from lib.config.schema import AugmentationConfig, BinarizerFeaturesConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.indexed_dataset import IndexedDataset
from lib.sequence_mutation import apply_mask_mutations, apply_sequence_edits
from lib.vocabulary import MASK_TOKEN, SPACE_TOKEN, NUM_RESERVED_TOKENS, Vocabulary
from .augmentation import (
    AugmentationContext,
    ComposedAugmentation,
    SpectrogramStretching,
    generate_seed,
    build_augmentation_chain,
)

__all__ = [
    "collate_nd",
    "BaseDataset",
    "PhonemeTimingDataset",
    "TextOnlyDataset",
    "DynamicBatchSampler",
    "ZippedDataLoader",
]


def _align_length(t: torch.Tensor, expected: int) -> torch.Tensor:
    """Slice or pad *t* along dim 0 to exactly *expected* frames."""
    n = t.shape[0]
    if n > expected:
        return t[:expected]
    if n < expected:
        return torch.nn.functional.pad(t, (0, 0, 0, expected - n), value=math.log(1e-5))
    return t


def collate_nd(values, pad_value=0, max_len=None, ndim=1):
    """
    Pad a list of Nd tensors on their first ``ndim`` dimensions and stack them
    into a (N+1)d tensor.
    """
    max_sizes = [
        max(v.size(d) for v in values)
        for d in range(ndim)
    ]
    if max_len is not None:
        max_sizes[0] = max_len
    remaining = values[0].shape[ndim:]
    size = (*max_sizes, *remaining)
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        idx = [i] + [slice(v.size(d)) for d in range(ndim)]
        res[tuple(idx)] = v
    return res


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    __non_zero_paddings__ = {
        "spectrogram": math.log(1e-5),
        "spectrogram_dirty": math.log(1e-5),
    }
    __multi_dims__: dict[str, int] = {}

    def __init__(
            self,
            data_dir: pathlib.Path,
            prefix: str,
            *,
            augmentation_config: AugmentationConfig = None,
            augmentation_deterministic: bool = False,
            augmentation_return_dirty: bool = False,
            max_concat_size: int | None = None,
            max_concat_frames: int | None = None,
            concat_deterministic: bool = False,
            return_waveform: bool = False,
    ):
        super().__init__()
        if return_waveform and augmentation_config is not None:
            raise ValueError(
                "return_waveform is incompatible with augmentations: "
                "spectrogram-domain transforms cannot be reversed to waveform."
            )
        self.info = {
            k: v
            for k, v in numpy.load(data_dir / f"{prefix}.info.npz").items()
        }
        self.data_dir = data_dir
        self.data = IndexedDataset(data_dir, prefix)
        self.epoch = torch.multiprocessing.Value("i", 0)
        self.augmentation_config = augmentation_config
        self.augmentation_deterministic = augmentation_deterministic
        self.augmentation_return_dirty = augmentation_return_dirty
        self.augmentation_chains: dict[int, ComposedAugmentation] = {}
        self.mel_spectrogram = None
        self.max_concat_size = max_concat_size
        self.max_concat_frames = max_concat_frames
        self.concat_deterministic = concat_deterministic
        self.return_waveform = return_waveform
        self._n_original = len(self.info["lengths"])
        self._group_indices: list[list[int]] | None = None
        self._setup()
        if self.max_concat_size is not None or self.max_concat_frames is not None:
            self._form_groups(0)

    def __getitem__(self, index):
        if self._group_indices is not None:
            samples = [self._get_single_item(i) for i in self._group_indices[index]]
            result = self.concat_samples(samples)
            result["_idx"] = torch.tensor(index, dtype=torch.long)
            result["_augmentation"] = {}
            return result
        return self._get_single_item(index)

    def __len__(self):
        if self._group_indices is not None:
            return len(self._group_indices)
        return self._n_original

    def set_epoch(self, epoch: int):
        self.epoch.value = epoch
        if self.max_concat_size is not None or self.max_concat_frames is not None:
            self._form_groups(epoch)
        if self.augmentation_config is not None and not self.augmentation_deterministic:
            self._build_chains(numpy.random.default_rng())

    def num_frames(self, index: int) -> int:
        if self._group_indices is not None:
            return sum(
                self._single_num_frames(i) for i in self._group_indices[index]
            )
        return self._single_num_frames(index)

    def _single_num_frames(self, index: int) -> int:
        base_len = int(self.info["lengths"][index])
        chain = self.augmentation_chains.get(index)
        if chain is not None:
            for t in chain.transforms:
                if isinstance(t, SpectrogramStretching) and t.speed is not None:
                    return max(1, int(base_len / t.speed))
        return base_len

    def get_metadata(self, key: str, index: int):
        """Proxy for info[key][index] that handles group-to-original mapping."""
        if self._group_indices is None:
            return self.info[key][index]

        values = [self.info[key][orig_idx] for orig_idx in self._group_indices[index]]
        if key in ("item_paths", "item_texts"):
            return " ".join(str(v) for v in values)
        return sum(int(v) for v in values)

    @abc.abstractmethod
    def concat_samples(self, samples: list[dict]) -> dict:
        """Merge individual sample dicts into one concatenated sample.

        Each dict is the output of _get_single_item. samples is non-empty.
        When len(samples) == 1 the result should be the sample unchanged.
        """

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

    def _form_groups(self, epoch: int) -> None:
        seed = 42 if self.concat_deterministic else 42 + epoch
        rng = random.Random(seed)
        indices = list(range(self._n_original))
        rng.shuffle(indices)

        groups = []
        current = []
        current_frames = 0
        for idx in indices:
            frames = self._single_num_frames(idx)
            exceed_size = (
                    self.max_concat_size is not None
                    and len(current) >= self.max_concat_size
            )
            exceed_frames = (
                    self.max_concat_frames is not None
                    and current_frames + frames > self.max_concat_frames
            )
            if current and (exceed_size or exceed_frames):
                groups.append(current)
                current = []
                current_frames = 0
            current.append(idx)
            current_frames += frames
        if current:
            groups.append(current)

        self._group_indices = groups

    def _build_chains(self, generator: numpy.random.Generator):
        self.augmentation_chains.clear()
        for index in range(self._n_original):
            self.augmentation_chains[index] = build_augmentation_chain(
                self.augmentation_config,
                mel_spectrogram=self.mel_spectrogram,
                generator=generator,
            )

    def _get_single_item(self, index):
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
        expected_len = int(self.info["lengths"][index])
        spectrogram = _align_length(spectrogram, expected_len)

        if self.augmentation_config is not None and self.augmentation_return_dirty:
            spectrogram_clean = torch.clamp(spectrogram_clean, min=math.log(1e-5))
            spectrogram_clean = _align_length(spectrogram_clean, expected_len)
            sample["spectrogram"] = spectrogram_clean
            sample["spectrogram_dirty"] = spectrogram
        else:
            sample["spectrogram"] = spectrogram

        if "f0" in sample:
            f0_val = sample["f0"].float()
            if f0_val.shape[0] != spectrogram.shape[0]:
                f0_val = torch.nn.functional.interpolate(
                    f0_val.view(1, 1, -1), size=spectrogram.shape[0],
                    mode="linear", align_corners=True,
                ).view(-1)
            sample["f0"] = f0_val

        if self.return_waveform:
            sample["waveform"] = torch.from_numpy(waveform).float()
            sample["duration"] = torch.tensor(
                waveform.shape[0] / self.sample_rate, dtype=torch.float32,
            )
        return {
            "_idx": torch.tensor(index, dtype=torch.long),
            "_augmentation": augmentation,
            **sample,
        }

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
            "indices": torch.stack([s.pop("_idx") for s in samples]),
        }
        if len(samples) == 0:
            return batch
        for key, value in samples[0].items():
            if value.ndim == 0:
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                pad_value = cls.__non_zero_paddings__.get(key, 0)
                ndim = cls.__multi_dims__.get(key, 1)
                batch[key] = collate_nd([s[key] for s in samples], pad_value=pad_value, ndim=ndim)
        return batch


class PhonemeTimingDataset(BaseDataset):
    def __init__(
            self,
            *args,
            augmentation_return_mutated: bool = False,
            augmentation_config=None,
            **kwargs,
    ):
        # Time stretching is not supported for token-spectrogram aligned datasets.
        if augmentation_config is not None:
            augmentation_config = augmentation_config.drop("time_stretching")
        super().__init__(
            *args,
            augmentation_config=augmentation_config,
            **kwargs,
        )
        self._augmentation_return_mutated = augmentation_return_mutated
        self._vocab_size: int | None = None
        if (
                self.augmentation_config is not None
                and self.augmentation_config.sequence_edit.enabled
        ):
            self._vocab_size = Vocabulary.from_file(
                self.data_dir / "vocabulary.json"
            ).vocab_size

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        if self._group_indices is not None:
            return sample  # already processed by concat_samples
        return self._prepare_item(sample)

    def _prepare_item(self, sample: dict) -> dict:
        # Per-frame ground-truth token IDs, computed before any token edits.
        regions_orig = sample["regions"]
        tokens_orig = sample["tokens"]
        T = int(regions_orig.shape[0])
        frame_targets = torch.zeros(T, dtype=torch.long)
        valid = regions_orig > 0
        if valid.any():
            idx = regions_orig[valid] - 1
            frame_targets[valid] = tokens_orig[idx]
        sample["frame_targets"] = frame_targets

        rng = random.Random(sample["_idx"].item()) if self.augmentation_deterministic else None
        _rand = rng or random

        # Determine whether and how to mutate
        edit_cfg = self.augmentation_config and self.augmentation_config.sequence_edit
        mask_cfg = self.augmentation_config and self.augmentation_config.token_masking
        mutation_type = None  # "edit", "mask", or None (identity)
        if self._vocab_size is not None and _rand.random() < edit_cfg.prob:
            mutation_type = "edit"
        elif mask_cfg is not None and mask_cfg.enabled and _rand.random() < mask_cfg.prob:
            mutation_type = "mask"

        if self._augmentation_return_mutated:
            # Keep originals clean; emit mutated copy
            sample["token_targets"] = sample["tokens"].clone()
            sample["is_mlm"] = torch.tensor(False, dtype=torch.bool)
            if mutation_type == "edit":
                mutated_tok, _, _, mutated_targets = apply_sequence_edits(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    min_token=NUM_RESERVED_TOKENS,
                    max_token=self._vocab_size - 1,
                    p_sub=edit_cfg.p_sub,
                    p_del=edit_cfg.p_del,
                    p_ins=edit_cfg.p_ins,
                    rng=rng,
                )
                sample["tokens_mutated"] = mutated_tok
                sample["token_targets_mutated"] = mutated_targets
            elif mutation_type == "mask":
                mutated_tok, _, _, mutated_targets = apply_mask_mutations(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    p_mask=mask_cfg.p_mask,
                    p_insert=mask_cfg.p_insert,
                    p_chain=mask_cfg.p_chain,
                    max_chain=mask_cfg.max_chain,
                    mask_token=MASK_TOKEN,
                    space_token=SPACE_TOKEN,
                    rng=rng,
                )
                sample["tokens_mutated"] = mutated_tok
                sample["token_targets_mutated"] = mutated_targets
            else:
                sample["tokens_mutated"] = sample["tokens"].clone()
                sample["token_targets_mutated"] = sample["tokens"].clone()
        else:
            # Mutate in-place
            if mutation_type == "edit":
                (
                    sample["tokens"], sample["spans"], sample["regions"],
                    sample["token_targets"],
                ) = apply_sequence_edits(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    min_token=NUM_RESERVED_TOKENS,
                    max_token=self._vocab_size - 1,
                    p_sub=edit_cfg.p_sub,
                    p_del=edit_cfg.p_del,
                    p_ins=edit_cfg.p_ins,
                    rng=rng,
                )
                sample["is_mlm"] = torch.tensor(False, dtype=torch.bool)
            elif mutation_type == "mask":
                (
                    sample["tokens"], sample["spans"], sample["regions"],
                    sample["token_targets"],
                ) = apply_mask_mutations(
                    tokens=sample["tokens"],
                    spans=sample["spans"],
                    regions=sample["regions"],
                    p_mask=mask_cfg.p_mask,
                    p_insert=mask_cfg.p_insert,
                    p_chain=mask_cfg.p_chain,
                    max_chain=mask_cfg.max_chain,
                    mask_token=MASK_TOKEN,
                    space_token=SPACE_TOKEN,
                    rng=rng,
                )
                sample["is_mlm"] = torch.tensor(True, dtype=torch.bool)
            else:
                sample["token_targets"] = sample["tokens"].clone()
                sample["is_mlm"] = torch.tensor(False, dtype=torch.bool)

        sample["T"] = torch.tensor(sample["spectrogram"].shape[0], dtype=torch.long)
        sample["N"] = torch.tensor(sample["tokens"].shape[0], dtype=torch.long)
        return sample

    def concat_samples(self, samples: list[dict]) -> dict:
        if len(samples) == 1:
            return self._prepare_item(samples[0])

        processed = [self._prepare_item(s) for s in samples]

        specs = [s["spectrogram"] for s in processed]
        T_cumsum = [0]
        for sp in specs[:-1]:
            T_cumsum.append(T_cumsum[-1] + sp.shape[0])
        N_vals = [int(s["N"].item()) for s in processed]
        N_cumsum = [0]
        for nv in N_vals[:-1]:
            N_cumsum.append(N_cumsum[-1] + nv)

        shifted_spans = []
        for s, t_off in zip(processed, T_cumsum):
            shifted_spans.append(s["spans"] + t_off if t_off > 0 else s["spans"])

        shifted_regions = []
        for s, n_off in zip(processed, N_cumsum):
            r = s["regions"]
            if n_off > 0:
                r = r.clone()
                mask = r > 0
                r[mask] += n_off
            shifted_regions.append(r)

        result = {
            "spectrogram": torch.cat(specs, dim=0),
            "tokens": torch.cat([s["tokens"] for s in processed]),
            "spans": torch.cat(shifted_spans),
            "regions": torch.cat(shifted_regions),
            "token_targets": torch.cat([s["token_targets"] for s in processed]),
            "frame_targets": torch.cat([s["frame_targets"] for s in processed]),
            "is_mlm": torch.any(torch.stack([s["is_mlm"] for s in processed])),
            "T": torch.tensor(sum(s["T"].item() for s in processed)),
            "N": torch.tensor(sum(N_vals)),
        }
        if "tokens_mutated" in processed[0]:
            result["tokens_mutated"] = torch.cat([s["tokens_mutated"] for s in processed])
            result["token_targets_mutated"] = torch.cat([s["token_targets_mutated"] for s in processed])
        if "f0" in processed[0]:
            result["f0"] = torch.cat([s["f0"] for s in processed], dim=0)
        if "waveform" in processed[0]:
            result["waveform"] = torch.cat([s["waveform"] for s in processed], dim=0)
            result["duration"] = torch.stack([s["duration"] for s in processed]).sum()
        return result


class TextOnlyDataset(BaseDataset):
    __multi_dims__ = {
        **BaseDataset.__multi_dims__,
        "paths": 2
    }
    __non_zero_paddings__ = {
        **BaseDataset.__non_zero_paddings__,
        "widths": 1,
    }

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        if self._group_indices is not None:
            return sample  # already processed by concat_samples
        return self._prepare_item(sample)

    # noinspection PyMethodMayBeStatic
    def _prepare_item(self, sample: dict) -> dict:
        sample["T"] = torch.tensor(sample["spectrogram"].shape[0], dtype=torch.long)
        sample["N"] = torch.tensor(sample["paths"].shape[0], dtype=torch.long)
        return sample

    def concat_samples(self, samples: list[dict]) -> dict:
        if len(samples) == 1:
            return self._prepare_item(samples[0])

        processed = [self._prepare_item(s) for s in samples]

        specs = [s["spectrogram"] for s in processed]
        merged_spec = torch.cat(specs, dim=0)

        S_vals = [int(s["segments"].max().item()) for s in processed]
        S_cumsum = [0]
        for sv in S_vals[:-1]:
            S_cumsum.append(S_cumsum[-1] + sv)

        segs = []
        for s, offset in zip(processed, S_cumsum):
            seg = s["segments"]
            if offset > 0:
                seg = seg.clone()
                mask = seg > 0
                seg[mask] += offset
            segs.append(seg)
        merged_segments = torch.cat(segs)

        merged_widths = torch.cat([s["widths"] for s in processed])

        max_width = max(s["paths"].shape[1] for s in processed)
        padded = []
        for s in processed:
            pw = s["paths"].shape[1]
            if pw < max_width:
                p = torch.nn.functional.pad(s["paths"], (0, max_width - pw))
            else:
                p = s["paths"]
            padded.append(p)
        merged_paths = torch.cat(padded, dim=0)

        T_vals = [s["T"].item() for s in processed]
        N_vals = [s["N"].item() for s in processed]

        result = {
            "spectrogram": merged_spec,
            "paths": merged_paths,
            "segments": merged_segments,
            "widths": merged_widths,
            "T": torch.tensor(sum(T_vals)),
            "N": torch.tensor(sum(N_vals)),
        }
        if "f0" in processed[0]:
            result["f0"] = torch.cat([s["f0"] for s in processed], dim=0)
        if "waveform" in processed[0]:
            result["waveform"] = torch.cat([s["waveform"] for s in processed], dim=0)
            result["duration"] = torch.stack([s["duration"] for s in processed]).sum()
        return result


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
            target_num_batches: int | None = None,
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
        self.target_num_batches = target_num_batches

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

        total_items = len(sorted_indices)
        if self.target_num_batches is not None:
            target_items_per_batch = max(1, math.ceil(total_items / self.target_num_batches))
            effective_max_batch_size = min(self.max_batch_size, target_items_per_batch)
        else:
            effective_max_batch_size = self.max_batch_size

        def batch_full(batch_: list[int], new_index_: int):
            if len(batch_) >= effective_max_batch_size:
                return True
            max_len = max(lengths[new_index_], max((lengths[i] for i in batch_), default=0))
            if max_len * (len(batch_) + 1) > self.max_batch_frames:
                return True
            return False

        def _greedy_pack() -> list[list[int]]:
            batches_: list[list[int]] = []
            current_batch = []
            for _idx in sorted_indices:
                sample_length = lengths[_idx]
                if sample_length > self.max_batch_frames:
                    raise ValueError(
                        f"Sample length {sample_length} exceeds max batch frames {self.max_batch_frames}."
                    )
                if batch_full(current_batch, _idx):
                    batches_.append(current_batch)
                    current_batch = []
                current_batch.append(_idx)
            if current_batch:
                batches_.append(current_batch)
            return batches_

        batches = _greedy_pack()

        if self.target_num_batches is not None:
            # Merge: too many batches  --  merge the smallest adjacent pairs
            while len(batches) > self.target_num_batches:
                best_i = -1
                best_size = float("inf")
                for i in range(len(batches) - 1):
                    combined = len(batches[i]) + len(batches[i + 1])
                    if combined > self.max_batch_size:
                        continue
                    combined_len = max(
                        max((lengths[j] for j in batches[i]), default=0),
                        max((lengths[j] for j in batches[i + 1]), default=0),
                    )
                    if combined_len * combined > self.max_batch_frames:
                        continue
                    if combined < best_size:
                        best_size = combined
                        best_i = i
                if best_i < 0:
                    break  # no more mergeable pairs
                batches[best_i].extend(batches[best_i + 1])
                del batches[best_i + 1]

            # Reduce: too few batches  --  halve effective size and repack
            while len(batches) < self.target_num_batches:
                if effective_max_batch_size <= 1:
                    raise RuntimeError(
                        f"Cannot form {self.target_num_batches} batches from "
                        f"{total_items} items: aux dataset too small. "
                        f"Reduce aux_multiplier or add more data."
                    )
                effective_max_batch_size = max(1, effective_max_batch_size // 2)
                batches = _greedy_pack()

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


class ZippedDataLoader:
    """
    Wraps a main DataLoader, optionally zipped with an aux DataLoader.
    Batches are always wrapped as ``{"main": ..., "size": ...}``.
    When aux is present, ``"aux"`` key is included.
    Epoch length is defined by the main DataLoader.
    """

    def __init__(
            self,
            main_dataloader: torch.utils.data.DataLoader,
            aux_dataloader: torch.utils.data.DataLoader | None = None,
            aux_sampler: DynamicBatchSampler | None = None,
    ):
        self.main_dl = main_dataloader
        self.aux_dl = aux_dataloader
        self.aux_sampler = aux_sampler

    def __iter__(self):
        main_iter = iter(self.main_dl)

        if self.aux_dl is not None:
            n_batches = len(self.main_dl)
            self.aux_sampler.target_num_batches = n_batches
            self.aux_sampler.formed = None
            aux_iter = iter(self.aux_dl)

            for main_batch in main_iter:
                try:
                    aux_batch = next(aux_iter)
                except StopIteration:
                    self.aux_sampler.formed = None
                    aux_iter = iter(self.aux_dl)
                    aux_batch = next(aux_iter)
                yield {
                    "main": main_batch,
                    "aux": aux_batch,
                    "size": main_batch["size"],
                }
        else:
            for main_batch in main_iter:
                yield {
                    "main": main_batch,
                    "size": main_batch["size"],
                }

    def __len__(self):
        return len(self.main_dl)
