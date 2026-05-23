"""Paradigm base classes for G2P converters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from g2p.converters.base import Converter, PronunciationGroup


class PronunciationScriptConverter(Converter, ABC):
    """G2P for writing systems that use a decoupled *pronunciation script*.

    Two-phase convert:

    1. ``text_to_script`` — text tokens are rendered into pronunciation-
       script tokens (pinyin, jyutping, romaji, …).
    2. ``script_to_phonemes`` — each script token is mapped to one or
       more phoneme sequences (typically via dictionary lookup).
    """

    @abstractmethod
    def text_to_script(self, tokens: list[str]) -> list[list[str]]:
        """Convert text tokens to pronunciation-script tokens.

        Each inner list holds the alternative script representations for
        one input token.  A single-reading token has a one-element inner
        list.
        """
        ...

    @abstractmethod
    def script_to_phonemes(self, script: str) -> list[list[str]]:
        """Map a single script token to its phoneme sequences."""
        ...

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        scripts_per_token = self.text_to_script(tokens)
        result: list[PronunciationGroup] = []
        for scripts in scripts_per_token:
            seen: set[tuple[str, ...]] = set()
            paths: list[list[str]] = []
            for s in scripts:
                for phonemes in self.script_to_phonemes(s):
                    key = tuple(phonemes)
                    if key not in seen:
                        seen.add(key)
                        paths.append(list(phonemes))
            result.append(PronunciationGroup(paths=paths))
        return result
