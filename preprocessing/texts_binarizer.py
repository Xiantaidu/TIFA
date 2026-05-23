import csv
import pathlib
from dataclasses import dataclass

import numpy

from g2p.api import build_pipeline_from_config
from g2p.converters.base import PronunciationGroup
from lib import logging
from lib.levenshtein import align_multipath, merge_shared

from .binarizer_base import (
    BaseBinarizer,
    DataSample,
    MetadataItem,
    find_waveform_file,
)

TEXTS_ITEM_ATTRIBUTES = [
    "tokens",  # [T, max_width] int64 — path grid, 0 = no token
    "partitions",  # [T] int64 — 1-based partition index per grid position
    "widths",  # [P] int64 — number of alternative sub-paths per partition
]


@dataclass
class TextMetadataItem(MetadataItem):
    text: str
    pronunciations: list[PronunciationGroup] | None = None


class TextOnlyBinarizer(BaseBinarizer):
    __data_attrs__ = TEXTS_ITEM_ATTRIBUTES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.g2p is None:
            raise RuntimeError(
                "G2P pipeline is not configured but TextOnlyBinarizer requires it."
            )
        self.g2p = build_pipeline_from_config(self.config.g2p)

    def resolve_data_dir(self) -> pathlib.Path:
        return self.config.text_only_data_dir_resolved

    def load_metadata(self, subset_dir: pathlib.Path) -> list[MetadataItem]:
        index_path = subset_dir / "index.csv"
        with open(index_path, "r", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        items: list[MetadataItem] = []
        for row in rows:
            name = row["name"]
            waveform_fn = find_waveform_file(subset_dir, name)
            if waveform_fn is None:
                continue
            language = row["language"]
            text = row["text"]
            try:
                groups = self.g2p.convert(text, languages=[language])
            except Exception as e:
                logging.warning(
                    f"G2P failed for item '{name}': {e}"
                )
                continue
            symbols: list[str] = []
            for pg in groups:
                for path in pg.paths:
                    symbols.extend(path)
            estimated_duration = (
                self.get_frame_count(waveform_fn) * self.timestep
            )
            items.append(TextMetadataItem(
                name=name,
                language=language,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                raw_symbols=symbols,
                text=text,
                pronunciations=groups,
            ))
        return items

    def process_item(self, item: TextMetadataItem) -> DataSample:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary has not been built.")

        length = self.get_frame_count(item.waveform_fn)

        groups = item.pronunciations
        if groups is None:
            raise RuntimeError(f"G2P not run for item '{item.name}'")

        all_partitions: list[list[list[int]]] = []
        for pg in groups:
            encoded_paths = []
            for path in pg.paths:
                tok_ids = []
                for ph in path:
                    tid = self.vocabulary.encode(ph, item.language)
                    if tid is None:
                        raise RuntimeError(
                            f"Token '{ph}' not in vocabulary "
                            f"for item '{item.name}'."
                        )
                    tok_ids.append(tid)
                encoded_paths.append(tok_ids)
            all_partitions.extend(align_multipath(encoded_paths))

        all_partitions = merge_shared(all_partitions)

        T = sum(
            max(len(sp) for sp in sub_paths)
            for sub_paths in all_partitions
        )
        max_width = max(
            (len(sub_paths) for sub_paths in all_partitions),
            default=0,
        )
        P = len(all_partitions)

        tokens = numpy.zeros((T, max_width), dtype=numpy.int64)
        partitions_arr = numpy.zeros(T, dtype=numpy.int64)
        widths = numpy.zeros(P, dtype=numpy.int64)

        col = 0
        for p_idx, sub_paths in enumerate(all_partitions):
            L = max(len(sp) for sp in sub_paths)
            widths[p_idx] = len(sub_paths)
            partitions_arr[col:col + L] = p_idx + 1
            for alt_idx, path in enumerate(sub_paths):
                for pos, tok_id in enumerate(path):
                    tokens[col + pos, alt_idx] = tok_id
            col += L

        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.name,
            length=length,
            text=item.text,
            data={
                "tokens": tokens,
                "partitions": partitions_arr,
                "widths": widths,
            },
        )
