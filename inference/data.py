import pathlib
from dataclasses import dataclass
from typing import Any, Literal

import librosa
import numpy
import textgrid
import torch
import torch.utils.data

from g2p.api import build_pipeline_from_config
from lib import logging
from lib.audio import load_audio
from lib.config.schema import G2PPipelineConfig
from lib.levenshtein import segment_groups
from lib.vocabulary import Vocabulary
from training.data import collate_nd


def _skip(identifier: str, reason: str) -> dict[str, Any]:
    return {"skip": True, "identifier": identifier, "warning": f"Skipping '{identifier}': {reason}"}


@dataclass(eq=False)
class _TokenWithWord:
    """Token with piggybacked group index and phoneme string for
    Levenshtein alignment.

    ``__eq__`` and ``__hash__`` use only *token*, so the alignment sees
    phoneme identity.  *group* and *phoneme* survive all Levenshtein
    operations unchanged.
    """

    token: int
    group: int
    phoneme: str

    def __eq__(self, other):
        if not isinstance(other, _TokenWithWord):
            return NotImplemented
        return self.token == other.token

    def __hash__(self):
        return hash(self.token)


def _deduplicate_paths(paths: list[list[_TokenWithWord]]) -> list[list[_TokenWithWord]]:
    seen: set[tuple[int, ...]] = set()
    result: list[list[_TokenWithWord]] = []
    for path in paths:
        key = tuple(tw.token for tw in path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _merge_segment_run(
        segments: list[list[list[_TokenWithWord]]],
) -> list[list[_TokenWithWord]]:
    merged: list[list[_TokenWithWord]] = [[]]
    for sub_paths in segments:
        merged = [prefix + path for prefix in merged for path in sub_paths]
    return _deduplicate_paths(merged)


def _merge_ambiguous_runs(
        segments: list[list[list[_TokenWithWord]]],
) -> list[list[list[_TokenWithWord]]]:
    result: list[list[list[_TokenWithWord]]] = []
    run: list[list[list[_TokenWithWord]]] = []

    for sub_paths in segments:
        if len(sub_paths) > 1:
            run.append(sub_paths)
            continue
        if run:
            result.append(_merge_segment_run(run))
            run = []
        result.append(sub_paths)

    if run:
        result.append(_merge_segment_run(run))

    return result


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
        path_grid_unit: Literal["levenshtein", "word"] = "levenshtein",
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
        self.path_grid_unit = path_grid_unit

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
            text_path = audio_path.with_suffix(".lab")
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

        # Build encoded_groups (one per G2PText). Phoneme tokens are
        # wrapped in _TokenWithWord so the G2PText index, original
        # phoneme string, and G2PWord index piggyback through
        # segment_groups unchanged.
        # lexicon: per G2PText, {word_label: [[phonemes, ...], ...]}
        # used by the callback to pick the best label for the chosen
        # phoneme sequence via Levenshtein distance.
        encoded_groups: list[list[list[_TokenWithWord]]] = []
        lexicon: list[dict[str, list[list[str]]]] = []
        warning = ""

        for gt_idx, gt in enumerate(g2p_texts):
            group_paths: list[list[_TokenWithWord]] = []
            gw_paths: dict[str, list[list[str]]] = {}
            oov_count = 0
            for gw in gt.words:
                for path in gw.phones:
                    tok_ids = [
                        self.vocabulary.encode(ph, gt.language) for ph in path
                    ]
                    if any(tid is None for tid in tok_ids):
                        oov_phs = [ph for tid, ph in zip(tok_ids, path) if tid is None]
                        if self.oov_handling == "raise":
                            return {"skip": True, "error": f"Unknown phonemes {oov_phs} in text for '{identifier}'"}
                        if self.oov_handling == "skip":
                            return _skip(identifier, f"Unknown phoneme {oov_phs} in text")
                        # forced: drop this path
                        oov_count += 1
                        continue
                    group_paths.append([
                        _TokenWithWord(tid, gt_idx + 1, ph)
                        for tid, ph in zip(tok_ids, path)
                    ])
                    gw_paths.setdefault(gw.word, []).append(path)

            if oov_count > 0:
                warning = f"Dropped {oov_count} OOV pronunciation(s)"

            if not group_paths:
                continue
            encoded_groups.append(group_paths)
            lexicon.append(gw_paths)

        if not encoded_groups:
            return _skip(identifier, "No valid token sequence")

        if self.path_grid_unit == "levenshtein":
            all_segments = segment_groups(encoded_groups)
        elif self.path_grid_unit == "word":
            word_segments = [_deduplicate_paths(group_paths) for group_paths in encoded_groups]
            all_segments = [
                sub_paths for sub_paths in _merge_ambiguous_runs(word_segments)
                if any(len(path) > 0 for path in sub_paths)
            ]
        else:
            raise ValueError(f"Unknown grid unit: {self.path_grid_unit}")

        if not all_segments:
            return _skip(identifier, "No valid token sequence")

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
        groups_arr = numpy.zeros((N, max_width), dtype=numpy.int64)
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
                    groups_arr[col + pos, alt_idx] = tw.group
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
            "groups": torch.from_numpy(groups_arr).long(),
            "segments": torch.from_numpy(segments_arr).long(),
            "widths": torch.from_numpy(widths).long(),
            "phonemes": alts_to_phonemes,
            "lexicon": lexicon,
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
            "groups": collate_nd(
                [b["groups"] for b in valid], pad_value=0, ndim=2,
            ),
            "segments": collate_nd(
                [b["segments"] for b in valid], pad_value=0,
            ),
            "widths": collate_nd(
                [b["widths"] for b in valid], pad_value=1,
            ),
            "phonemes": [b["phonemes"] for b in valid],
            "lexicon": [b["lexicon"] for b in valid],
        }


class TextGridDataset(torch.utils.data.Dataset):
    """Parses TextGrid files from a directory recursively.

    Each item returns intervals from a named tier with empty marks filtered.
    """

    def __init__(
        self,
        directory: pathlib.Path,
        tier_name: str = "phones",
        stop_symbols: set[str] | None = None,
    ):
        self.directory = pathlib.Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        self.tier_name = tier_name
        self.stop_symbols = stop_symbols or set()
        self._files = sorted(
            p for p in self.directory.glob("**/*.TextGrid")
            if p.is_file()
        )
        if not self._files:
            raise FileNotFoundError(
                f"No .TextGrid files found in {directory}"
            )

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        filepath = self._files[index]
        identifier = str(
            filepath.relative_to(self.directory).with_suffix("")
        )
        tg = textgrid.TextGrid.fromFile(str(filepath))

        onsets: list[float] = []
        offsets: list[float] = []
        marks: list[str] = []

        for tier in tg:
            if tier.name == self.tier_name and isinstance(
                tier, textgrid.IntervalTier
            ):
                for interval in tier:
                    mark = interval.mark.strip()
                    if not mark or mark in self.stop_symbols:
                        continue
                    onsets.append(float(interval.minTime))
                    offsets.append(float(interval.maxTime))
                    marks.append(mark)
                break

        return {
            "identifier": identifier,
            "onsets": onsets,
            "offsets": offsets,
            "marks": marks,
        }


class PairedDataset(torch.utils.data.Dataset):
    """Pairs two datasets by identifier for offline evaluation.

    Accepts any two Datasets whose items contain ``"identifier"``,
    ``"onsets"``, ``"offsets"``, ``"marks"``.  Verifies label equality,
    builds an internal vocabulary, and encodes spans + tokens.
    """

    def __init__(
        self,
        pred_dataset: torch.utils.data.Dataset,
        gt_dataset: torch.utils.data.Dataset,
        mismatch_handling: Literal["raise", "skip"] = "raise",
    ):
        self.mismatch_handling = mismatch_handling
        self.vocab_size: int = 0

        pred_by_id = {item["identifier"]: item for item in pred_dataset}
        gt_by_id = {item["identifier"]: item for item in gt_dataset}

        pred_only = set(pred_by_id) - set(gt_by_id)
        gt_only = set(gt_by_id) - set(pred_by_id)
        for oid in sorted(pred_only):
            logging.warning(f"Prediction '{oid}' has no ground-truth counterpart, skipped")
        for oid in sorted(gt_only):
            logging.warning(f"Ground-truth '{oid}' has no prediction counterpart, skipped")

        common = sorted(set(pred_by_id) & set(gt_by_id))
        if not common:
            raise ValueError(
                "No matching identifiers found between pred and gt datasets"
            )

        self._items: list[dict] = []
        all_marks: set[str] = set()

        for identifier in common:
            P = pred_by_id[identifier]
            G = gt_by_id[identifier]

            p_marks = P["marks"]
            g_marks = G["marks"]

            if len(p_marks) != len(g_marks):
                msg = (
                    f"'{identifier}': pred has {len(p_marks)} intervals, "
                    f"gt has {len(g_marks)}"
                )
                if self.mismatch_handling == "raise":
                    raise ValueError(msg)
                logging.warning(f"Skipping {msg}")
                continue

            for i, (mp, mg) in enumerate(zip(p_marks, g_marks)):
                if mp != mg:
                    msg = (
                        f"'{identifier}'[{i}]: pred='{mp}' vs gt='{mg}'"
                    )
                    if self.mismatch_handling == "raise":
                        raise ValueError(msg)
                    logging.warning(f"Skipping {msg}")
                    break
            else:
                all_marks.update(p_marks)
                self._items.append({
                    "identifier": identifier,
                    "pred_onsets": P["onsets"],
                    "pred_offsets": P["offsets"],
                    "gt_onsets": G["onsets"],
                    "gt_offsets": G["offsets"],
                    "marks": p_marks,
                })

        if not self._items:
            raise ValueError("No valid pairs after verification")

        self._mark_to_id = {
            label: i + 3  # NUM_RESERVED_TOKENS
            for i, label in enumerate(sorted(all_marks))
        }
        self.vocab = Vocabulary(symbol_to_id=self._mark_to_id)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item = self._items[index]
        tokens = torch.tensor(
            [self._mark_to_id[m] for m in item["marks"]], dtype=torch.int64
        )
        spans_pred = torch.stack([
            torch.tensor(item["pred_onsets"], dtype=torch.float32),
            torch.tensor(item["pred_offsets"], dtype=torch.float32),
        ], dim=-1)
        spans_gt = torch.stack([
            torch.tensor(item["gt_onsets"], dtype=torch.float32),
            torch.tensor(item["gt_offsets"], dtype=torch.float32),
        ], dim=-1)
        return {
            "identifier": item["identifier"],
            "tokens": tokens,
            "spans": spans_gt,
            "spans_pred": spans_pred,
        }

    @staticmethod
    def collate(samples: list[dict]) -> dict:
        return {
            "identifier": [s["identifier"] for s in samples],
            "tokens": collate_nd([s["tokens"] for s in samples], pad_value=0, ndim=1),
            "spans": collate_nd([s["spans"] for s in samples], pad_value=0.0, ndim=2),
            "spans_pred": collate_nd(
                [s["spans_pred"] for s in samples], pad_value=0.0, ndim=2
            ),
        }
