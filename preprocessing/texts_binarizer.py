import csv
import pathlib
from dataclasses import dataclass

import librosa
import numpy

from g2p.api import build_pipeline_from_config
from g2p.converters.base import G2PText
from lib import logging
from lib.audio import load_audio
from lib.feature.pitch import get_pitch_parselmouth
from lib.levenshtein import segment_groups

from .binarizer_base import (
    BaseBinarizer,
    DataSample,
    MetadataItem,
    find_waveform_file,
)

TEXTS_ITEM_ATTRIBUTES = [
    "paths",  # [N, max(widths)] int64  --  path grid, 0 = no token
    "segments",  # [N] int64  --  1-based segment index per grid position
    "widths",  # [max(segments)] int64  --  number of alternative sub-paths per segment
    "f0",  # [T] float32, pitch in Hz
]


@dataclass
class TextMetadataItem(MetadataItem):
    text: str
    g2p_texts: list[G2PText] | None = None


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
                g2p_texts = self.g2p.convert(text, languages=[language])
            except Exception as e:
                logging.warning(
                    f"G2P failed for item '{name}': {e}"
                )
                continue
            symbols: list[str] = []
            for gt in g2p_texts:
                for gw in gt.words:
                    for path in gw.phones:
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
                g2p_texts=g2p_texts,
            ))
        return items

    def process_item(self, item: TextMetadataItem) -> DataSample:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary has not been built.")

        length = self.get_frame_count(item.waveform_fn)

        f0_cfg = self.config.features.f0
        f0 = None
        if f0_cfg.enabled:
            waveform, sr = load_audio(item.waveform_fn)
            if sr != self.config.features.audio_sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.config.features.audio_sample_rate)
            f0, _uv = get_pitch_parselmouth(
                waveform, self.config.features.audio_sample_rate, length,
                hop_size=self.config.features.hop_size,
                f0_min=f0_cfg.f0_min, f0_max=f0_cfg.f0_max,
            )

        groups = item.g2p_texts
        if groups is None:
            raise RuntimeError(f"G2P not run for item '{item.name}'")

        encoded_groups: list[list[list[int]]] = []
        for gt in groups:
            encoded_paths = []
            for gw in gt.words:
                for path in gw.phones:
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
            encoded_groups.append(encoded_paths)

        all_segments = segment_groups(encoded_groups)

        N = sum(
            max(len(sp) for sp in sub_paths)
            for sub_paths in all_segments
        )
        max_width = max(
            (len(sub_paths) for sub_paths in all_segments),
            default=0,
        )
        S = len(all_segments)

        paths = numpy.zeros((N, max_width), dtype=numpy.int64)
        segments_arr = numpy.zeros(N, dtype=numpy.int64)
        widths = numpy.zeros(S, dtype=numpy.int64)

        col = 0
        for s_idx, sub_paths in enumerate(all_segments):
            L = max(len(sp) for sp in sub_paths)
            widths[s_idx] = len(sub_paths)
            segments_arr[col:col + L] = s_idx + 1
            for alt_idx, path in enumerate(sub_paths):
                for pos, tok_id in enumerate(path):
                    paths[col + pos, alt_idx] = tok_id
            col += L

        data = {
            "paths": paths,
            "segments": segments_arr,
            "widths": widths,
        }
        if f0 is not None:
            data["f0"] = f0
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.name,
            length=length,
            text=item.text,
            data=data,
        )
