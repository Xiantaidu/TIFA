import pathlib
from dataclasses import dataclass
from typing import Any, Literal

import librosa
import numpy
import torch
import torch.utils.data

from g2p.api import build_pipeline_from_config
from lib.audio import load_audio
from lib.config.schema import G2PPipelineConfig
from lib.levenshtein import segment_groups
from lib.vocabulary import Vocabulary
from training.data import collate_nd


@dataclass(eq=False)
class _TokenWithWord:
    """Token with piggybacked word index and phoneme string for
    Levenshtein alignment.

    ``__eq__`` and ``__hash__`` use only *token*, so the alignment sees
    phoneme identity.  *word* and *phoneme* survive all Levenshtein
    operations unchanged.
    """

    token: int
    word: int
    phoneme: str

    def __eq__(self, other):
        if not isinstance(other, _TokenWithWord):
            return NotImplemented
        return self.token == other.token

    def __hash__(self):
        return hash(self.token)


class AudioTextDataset(torch.utils.data.Dataset):
    """Pairs audio files with text files for forced alignment inference.

    Takes a filemap ``{identifier: audio_path}``.  In ``__getitem__``
    the paired ``.txt`` file is located alongside the audio and G2P
    conversion produces phoneme path grids.
    """

    def __init__(
        self,
        filemap: dict[str, pathlib.Path],
        g2p_config: G2PPipelineConfig,
        g2p_root: str | pathlib.Path,
        vocabulary: Vocabulary,
        audio_sample_rate: int,
        language: str | set[str] | None = None,
        oov_handling: Literal["raise", "skip", "force"] = "skip",
    ):
        self.g2p_config = g2p_config
        self.g2p_root = g2p_root
        self._g2p_pipeline = None
        self.vocabulary = vocabulary
        self.sample_rate = audio_sample_rate
        if isinstance(language, str):
            language = [language]
        elif isinstance(language, set):
            language = list(language)
        self.language = language
        self.oov_handling = oov_handling

        self.items: list[tuple[pathlib.Path, str]] = [
            (audio_path, identifier)
            for identifier, audio_path in sorted(filemap.items())
        ]
        if not self.items:
            raise ValueError("Empty filemap")

    def _get_g2p(self):
        if self._g2p_pipeline is None:
            self._g2p_pipeline = build_pipeline_from_config(
                self.g2p_config, root_path=self.g2p_root,
            )
        return self._g2p_pipeline

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        g2p_pipeline = self._get_g2p()
        audio_path, identifier = self.items[idx]

        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            return _skip(identifier, "No paired text file")

        with open(text_path, "r", encoding="utf8") as f:
            text = f.read().strip()
        if not text:
            return _skip(identifier, "Empty text")

        try:
            g2p_texts = g2p_pipeline.convert(
                text, languages=self.language,
            )
        except Exception as e:
            return _skip(identifier, f"G2P failed: {e}")

        # Build encoded_groups (one per G2PText) and inverted index
        # from phonemes to words. Phoneme tokens are wrapped in
        # _TokenWithWord so the G2PText index and original phoneme
        # string piggyback through segment_groups unchanged.
        encoded_groups: list[list[list[_TokenWithWord]]] = []
        phonemes_to_words: list[dict[tuple[str, ...], str]] = []
        warning = ""

        for gt_idx, gt in enumerate(g2p_texts):
            group_paths: list[list[_TokenWithWord]] = []
            word_map: dict[tuple[str, ...], str] = {}
            oov_count = 0
            for gw in gt.words:
                for path in gw.phones:
                    tok_ids = [
                        self.vocabulary.encode(ph, gt.language) for ph in path
                    ]
                    if any(tid is None for tid in tok_ids):
                        if self.oov_handling == "raise":
                            return {"skip": True, "error": f"Unknown phoneme in text for '{identifier}'"}
                        if self.oov_handling == "skip":
                            return _skip(identifier, "Unknown phoneme in text")
                        # forced: drop this path
                        oov_count += 1
                        continue
                    group_paths.append([
                        _TokenWithWord(tid, gt_idx + 1, ph)
                        for tid, ph in zip(tok_ids, path)
                    ])
                    word_map[tuple(path)] = gw.word

            if oov_count > 0:
                warning = f"Dropped {oov_count} OOV pronunciation(s)"

            if not group_paths:
                continue
            encoded_groups.append(group_paths)
            phonemes_to_words.append(word_map)

        if not encoded_groups:
            return _skip(identifier, "No valid token sequence")

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
        words_arr = numpy.zeros((N, max_width), dtype=numpy.int64)
        segments_arr = numpy.zeros(N, dtype=numpy.int64)
        widths = numpy.zeros(S, dtype=numpy.int64)

        col = 0
        alts_to_phonemes: dict[tuple[int, int], list[str]] = {}
        for s_idx, sub_paths in enumerate(all_segments):
            L = max(len(sp) for sp in sub_paths)
            widths[s_idx] = len(sub_paths)
            segments_arr[col:col + L] = s_idx + 1
            for alt_idx, path in enumerate(sub_paths):
                alts_to_phonemes[(s_idx, alt_idx)] = [tw.phoneme for tw in path]
                for pos, tw in enumerate(path):
                    paths[col + pos, alt_idx] = tw.token
                    words_arr[col + pos, alt_idx] = tw.word
            col += L

        audio, sr = load_audio(audio_path)
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate,
            )

        return {
            "skip": False,
            "identifier": identifier,
            "warning": warning,
            "waveform": torch.from_numpy(audio).float(),
            "duration": len(audio) / self.sample_rate,
            "paths": torch.from_numpy(paths).long(),
            "words": torch.from_numpy(words_arr).long(),
            "segments": torch.from_numpy(segments_arr).long(),
            "widths": torch.from_numpy(widths).long(),
            "phonemes": alts_to_phonemes,
            "lexicon": phonemes_to_words,
        }

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [err for b in batch if (err := b.get("error"))]
        if errors:
            raise RuntimeError("\n".join(errors))

        valid = [b for b in batch if not b["skip"]]
        warning = [warn for b in batch if (warn := b.get("warning"))]

        if not valid:
            return {"warning": warning}

        return {
            "warning": warning,
            "identifier": [b["identifier"] for b in valid],
            "waveform": collate_nd(
                [b["waveform"] for b in valid], pad_value=0.,
            ),
            "duration": torch.tensor(
                [b["duration"] for b in valid], dtype=torch.float32,
            ),
            "paths": collate_nd(
                [b["paths"] for b in valid], pad_value=0, ndim=2,
            ),
            "words": collate_nd(
                [b["words"] for b in valid], pad_value=0, ndim=2,
            ),
            "segments": collate_nd(
                [b["segments"] for b in valid], pad_value=0,
            ),
            "widths": collate_nd(
                [b["widths"] for b in valid], pad_value=1,
            ),
            "lexicon": [b["lexicon"] for b in valid],
            "phonemes": [b["phonemes"] for b in valid],
        }


def _skip(identifier: str, reason: str) -> dict[str, Any]:
    return {"skip": True, "identifier": identifier, "warning": f"Skipping '{identifier}': {reason}"}
