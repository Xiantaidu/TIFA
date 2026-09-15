"""Encode complete G2P paths without losing their labels or provenance."""

from collections.abc import Iterable
from typing import Literal

import numpy as np

from lib.levenshtein import align_multiple_sequences

from lib.vocabulary import NUM_RESERVED_TOKENS, Vocabulary, is_stop_symbol
from .converters.base import G2PWord


class G2PEncodingError(Exception):
    """A word has no pronunciation encodable by the active vocabulary."""


def encode_paths(
        words: list[G2PWord],
        vocabulary: Vocabulary,
        oov_handling: Literal["raise", "discard", "force"] = "raise",
        *,
        default_language: str | None = None,
        global_symbols: Iterable[str] = (),
        stop_symbols: Iterable[str] = (),
) -> tuple[dict[str, np.ndarray], list[list[dict]], list[str]]:
    """Return numeric grids, candidate metadata and original word texts.

    Candidate columns remain intact across each word's aligned rows.
    The candidates mask distinguishes empty paths from absent candidates.
    Metadata contains reading indices, phoneme strings and group scripts.
    In force mode an OOV removes the entire candidate, never just a phone.
    Unencodable words raise G2PEncodingError; invalid arguments or G2P
    structures raise ValueError and must not be treated as OOV failures.
    """
    if oov_handling not in ("raise", "discard", "force"):
        raise ValueError(f"Unknown OOV handling: {oov_handling}")
    global_symbols = frozenset(global_symbols)
    stop_symbols = frozenset(stop_symbols)
    paths = []
    for word in words:
        language = word.language if word.language is not None else default_language
        candidates = []
        for reading_index, reading in enumerate(word.readings):
            for path in reading.paths:
                tokens, groups, phonemes, scripts = [], [], [], []
                unknown = []
                for group in path:
                    if not group.phonemes:
                        raise ValueError("An empty pronunciation must be an empty path, not an empty group.")
                    group_tokens, group_phonemes = [], []
                    for phoneme in group.phonemes:
                        if is_stop_symbol(phoneme, language, global_symbols, stop_symbols):
                            continue
                        token = vocabulary.encode(phoneme, language)
                        if token is None:
                            unknown.append(phoneme)
                            continue
                        if token < NUM_RESERVED_TOKENS:
                            raise ValueError(f"Reserved token in G2P output: {phoneme!r}")
                        group_tokens.append(token)
                        group_phonemes.append(f"{language}/{phoneme}")
                    if group_tokens:
                        scripts.append(group.script)
                        tokens.extend(group_tokens)
                        phonemes.extend(group_phonemes)
                        groups.extend([len(scripts)] * len(group_tokens))
                if unknown:
                    if oov_handling == "force":
                        continue
                    raise G2PEncodingError(f"Unknown phonemes {unknown} in word {word.text!r}")
                candidates.append({
                    "tokens": tokens,
                    "groups": groups,
                    "phonemes": phonemes,
                    "reading": reading_index,
                    "scripts": scripts,
                })
        if not candidates and oov_handling != "force":
            raise G2PEncodingError(f"No pronunciation paths for word {word.text!r}")
        paths.append(candidates)
    data = _pack_paths(paths)
    lexicon = [
        [{key: candidate[key] for key in ("reading", "phonemes", "scripts")}
         for candidate in candidates]
        for candidates in paths
    ]
    return data, lexicon, [word.text for word in words]


def _pack_paths(paths: list[list[dict]]) -> dict[str, np.ndarray]:
    """Lay out whole-word candidates without splitting or deduplicating them."""
    profiles = []
    for candidates in paths:
        rows = align_multiple_sequences([candidate["tokens"] for candidate in candidates])
        profile = np.asarray(
            [[token if token is not None else 0 for token in row] for row in rows],
            dtype=np.int64,
        ).reshape(len(candidates), -1) if candidates else np.zeros((0, 0), dtype=np.int64)
        profiles.append(profile)

    capacity = max(1, sum(profile.shape[1] for profile in profiles))
    width = max(1, max(map(len, paths), default=0))
    tokens = np.zeros((capacity, width), dtype=np.int64)
    groups = np.zeros_like(tokens)
    owners = np.zeros(capacity, dtype=np.int64)
    valid = np.zeros((len(paths), width), dtype=np.bool_)
    offset = 0
    for w, (candidates, profile) in enumerate(zip(paths, profiles)):
        size = profile.shape[1]
        count = len(candidates)
        valid[w, :count] = True
        owners[offset:offset + size] = w + 1
        tokens[offset:offset + size, :count] = profile.T
        for c, candidate in enumerate(candidates):
            present = profile[c] != 0
            groups[offset:offset + size, c][present] = candidate["groups"]
        offset += size
    return {"paths": tokens, "words": owners, "groups": groups, "candidates": valid}
