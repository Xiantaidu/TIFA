import re
from pathlib import Path

from g2p.registry import converter

from .base import Converter

_PRON_UNSAFE_RE = re.compile(r"\s*\(\d+\)$")


@converter(id="dictionary", language=None)
class DictionaryConverter(Converter):
    """Pronunciation dictionary lookup. Loads a tab-separated file:
    ``<word>\\t<ph1> <ph2> ...``. Duplicate words accumulate pronunciations;
    ``word(N)`` and ``word (N)`` suffixes are variant forms of the same word."""

    def __init__(self, path: str) -> None:
        self._dict: dict[str, list[list[str]]] = {}
        self._load(path)

    def _load(self, path: str) -> None:
        with open(Path(path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                word, _, phoneme_str = line.partition("\t")
                if not phoneme_str:
                    continue
                base_word = _PRON_UNSAFE_RE.sub("", word).lower()
                phonemes = phoneme_str.split()
                self._dict.setdefault(base_word, []).append(phonemes)

    def claim(self, token: str) -> bool:
        return token.lower() in self._dict

    def convert(self, tokens: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for token in tokens:
            pronunciations = self._dict.get(token.lower())
            if pronunciations is None:
                raise KeyError(
                    f"DictionaryConverter: token '{token}' not in dictionary. "
                    f"claim should have filtered it."
                )
            result.append(list(pronunciations[0]))
        return result


@converter(id="passthrough", language=None)
class PassthroughConverter(Converter):
    """Catch-all converter that returns each token as its own phoneme.
    Typically placed last in a chain as a fallback for unconverted tokens."""

    def claim(self, token: str) -> bool:
        return True

    def convert(self, tokens: list[str]) -> list[list[str]]:
        return [[t] for t in tokens]


@converter(id="char_phoneme", language=None)
class CharPhonemeConverter(Converter):
    """One-to-one character-to-phoneme mapping.
    Each character in a token is mapped to one or more phonemes."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def claim(self, token: str) -> bool:
        return all(c in self._mapping for c in token)

    def convert(self, tokens: list[str]) -> list[list[str]]:
        result: list[list[str]] = []
        for token in tokens:
            phonemes: list[str] = []
            for c in token:
                phonemes.extend(self._mapping[c])
            result.append(phonemes)
        return result
