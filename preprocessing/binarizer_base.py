import abc
import math
import pathlib
import random
from dataclasses import dataclass

import librosa
import matplotlib.pyplot as plt
import numpy
import tqdm

from lib import logging
from lib.config.io import save_raw_config
from lib.config.schema import BinarizerConfig
from lib.indexed_dataset import IndexedDatasetBuilder
from lib.multiprocess import FailedItem, chunked_multiprocess_run
from lib.plot import vocab_distribution_to_figure
from lib.vocabulary import Vocabulary, VocabularyBuilder

ACCEPTED_AUDIO_FORMATS = {".wav", ".flac", ".opus"}


@dataclass
class MetadataItem(abc.ABC):
    name: str
    language: str
    waveform_fn: pathlib.Path
    estimated_duration: float
    raw_symbols: list[str]  # including stop symbols


@dataclass
class DataSample:
    path: str
    name: str
    length: int
    text: str
    data: dict[str, int | float | numpy.ndarray]
    derived: dict[str, int] = None
    error: str = None

    def __post_init__(self):
        if self.derived is None:
            self.derived = {}


class BaseBinarizer(abc.ABC):
    __data_attrs__: list[str] = None

    def __init__(self, config: BinarizerConfig, eval_mode=False, aux_mode=False):
        self.config = config
        self.eval_mode = eval_mode
        self.aux_mode = aux_mode
        self.data_dir: pathlib.Path = self.resolve_data_dir()
        self.timestep = config.features.timestep

        self.vocabulary: Vocabulary | None = None
        self.vocab_builder: VocabularyBuilder | None = None

        self.valid_items: list[MetadataItem] = []
        self.train_items: list[MetadataItem] = []

    @abc.abstractmethod
    def resolve_data_dir(self) -> pathlib.Path:
        pass

    @abc.abstractmethod
    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        pass

    @abc.abstractmethod
    def process_item(self, item: MetadataItem) -> DataSample:
        pass

    def split_dataset(self, metadata_list: list[MetadataItem]):
        if self.aux_mode:
            self.train_items = sorted(
                metadata_list, key=lambda itm: itm.estimated_duration, reverse=True
            )
            return
        if self.eval_mode:
            # Put all items into validation set and leave training set empty.
            self.valid_items.extend(metadata_list)
            self.valid_items.sort(key=lambda itm: itm.estimated_duration, reverse=True)
        else:
            validation_indices = sorted(random.sample(
                range(len(metadata_list)),
                k=min(self.config.validation_count, len(metadata_list))
            ))
            validation_indices_set = set(validation_indices)
            for i, item in enumerate(metadata_list):
                if i in validation_indices_set:
                    self.valid_items.append(item)
                else:
                    self.train_items.append(item)
            self.train_items.sort(key=lambda itm: itm.estimated_duration, reverse=True)
        if not self.eval_mode and not self.train_items:
            raise RuntimeError("Training set is empty.")
        if not self.valid_items:
            raise RuntimeError("Validation set is empty.")

    def process_items(self, items: list[MetadataItem], prefix: str, multiprocessing=True):
        builder = IndexedDatasetBuilder(
            path=self.data_dir, prefix=prefix, allowed_attr=self.__data_attrs__
        )
        if multiprocessing and self.config.num_workers > 0:
            logging.debug(f"Processing {prefix} items with {self.config.num_workers} worker(s).")
            iterable = chunked_multiprocess_run(
                self.process_item, [(item,) for item in items], num_workers=self.config.num_workers
            )
        else:
            logging.debug(f"Processing {prefix} items in main process.")
            iterable = (self.process_item(item) for item in items)
        item_paths = []
        item_texts = []
        lengths = []
        attr_lengths = {}
        total_duration = 0
        with tqdm.tqdm(zip(items, iterable), total=len(items), desc=f"Processing {prefix} items") as progress:
            for item, sample in progress:
                if isinstance(sample, FailedItem):
                    logging.error(
                        f"Worker failed: {sample.exception}\n{sample.traceback_str}",
                        callback=progress.write
                    )
                    continue
                sample: DataSample
                if sample.error:
                    logging.error(
                        f"Error encountered in sample '{sample.name}': {sample.error}",
                        callback=progress.write
                    )
                    continue
                builder.add_item(sample.data)
                item_paths.append(sample.path)
                item_texts.append(sample.text)
                lengths.append(sample.length)
                for k, v in sample.data.items():
                    if isinstance(v, numpy.ndarray) and v.ndim > 0:
                        if k not in attr_lengths:
                            attr_lengths[k] = []
                        attr_lengths[k].append(v.shape[0])
                for k, v in sample.derived.items():
                    if k not in attr_lengths:
                        attr_lengths[k] = []
                    attr_lengths[k].append(v)
                duration = sample.length * self.timestep
                total_duration += duration
        builder.finalize()
        metadata = {
            "item_paths": item_paths,
            "item_texts": item_texts,
            "lengths": lengths,
            **attr_lengths
        }
        metadata = {
            k: numpy.array(v)
            for k, v in metadata.items()
        }
        with open(self.data_dir / f"{prefix}.info.npz", "wb") as f:
            numpy.savez(f, **metadata)

        logging.info(f"Total duration of {prefix}: {format_duration(total_duration)}.")
        logging.debug(f"Processing {prefix} items done.")

    def collect_metadata(self) -> list[MetadataItem]:
        index_file_paths = list(self.data_dir.rglob("index.csv"))
        metadata_list = []
        for index_file_path in index_file_paths:
            subset_dir = index_file_path.parent
            subset_metadata_list = self.load_metadata(subset_dir)
            metadata_list.extend(subset_metadata_list)
            logging.debug(f"Loaded {len(subset_metadata_list)} metadata items from '{subset_dir.as_posix()}'.")
        return metadata_list

    def build_vocabulary(self, metadata_list: list[MetadataItem]):
        vocab_builder = VocabularyBuilder(
            global_symbols=self.config.vocabulary.global_symbols,
            stop_symbols=self.config.vocabulary.stop_symbols,
            merged_groups=self.config.vocabulary.merged_groups,
            peers=self.config.vocabulary.peers,
        )
        for item in metadata_list:
            vocab_builder.add(item.raw_symbols, default_language=item.language)
        self.vocab_builder = vocab_builder
        self.vocabulary = vocab_builder.build()
        self.save_vocab_plot(vocab_builder.counter())

    def save_vocab_plot(self, counter) -> None:
        fig = vocab_distribution_to_figure(counter)
        if fig is not None:
            filename = self.data_dir / "vocab_distribution.jpg"
            fig.savefig(fname=filename, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            logging.info(f"Vocabulary distribution plot saved to '{filename.as_posix()}'.")

    def build_dataset(self, metadata_list: list[MetadataItem]):
        self.split_dataset(metadata_list)
        logging.info(f"Training set total size: {len(self.train_items)}.")
        logging.info(f"Validation set total size: {len(self.valid_items)}.")
        self._process_datasets()

    def save_auxiliary_files(self):
        save_raw_config(self.config.features.model_dump(), self.data_dir / "feature.yaml")
        self.vocabulary.dump(self.data_dir / "vocabulary.json")
        self.vocab_builder.dump_token_peers(self.data_dir / "token_peers.json")

    def _process_datasets(self):
        if self.aux_mode:
            self.process_items(self.train_items, prefix="aux", multiprocessing=True)
        elif self.eval_mode:
            self.process_items(self.valid_items, prefix="valid", multiprocessing=True)
        else:
            self.process_items(self.valid_items, prefix="valid", multiprocessing=False)
            self.process_items(self.train_items, prefix="train", multiprocessing=True)

    def process(self):
        metadata_list = self.collect_metadata()
        self.build_vocabulary(metadata_list)
        self.build_dataset(metadata_list)
        self.save_auxiliary_files()

    def get_frame_count(self, waveform_fn) -> int:
        duration = librosa.get_duration(path=waveform_fn)
        return max(1, math.ceil(duration / self.timestep))

    def sec_dur_to_frame_dur(self, dur_sec: numpy.ndarray, length: int) -> numpy.ndarray:
        dur_cumsum = numpy.round(numpy.cumsum(dur_sec, axis=0) / self.timestep).astype(numpy.int64)
        dur_cumsum = numpy.clip(dur_cumsum, a_min=0, a_max=length)
        dur_cumsum[-1] = length
        return numpy.diff(dur_cumsum, axis=0, prepend=numpy.array([0]))


def find_waveform_file(subset_dir: pathlib.Path, item_name: str) -> pathlib.Path:
    for ext in ACCEPTED_AUDIO_FORMATS:
        searched_wav_fn = subset_dir / "waveforms" / f"{item_name}{ext}"
        if searched_wav_fn.exists():
            return searched_wav_fn
    logging.error(
        f"Waveform file missing in raw dataset \'{subset_dir.as_posix()}\': "
        f"item {item_name}, searched extensions: {', '.join(ACCEPTED_AUDIO_FORMATS)}."
    )
    return None


def format_duration(seconds: float) -> str:
    """Formats a duration in seconds to a 'XhYmZs' string."""
    if seconds < 0:
        raise ValueError("Duration cannot be negative.")

    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")

    if sec > 0:
        if parts:
            parts.append(f"{int(sec)}s")
        else:
            parts.append(f"{sec:.2f}s")
    elif not parts:
        return "0s"

    return "".join(parts)
