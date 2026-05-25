"""Paradigm base classes for G2P converters."""

from abc import ABC, abstractmethod

from g2p.converters.base import Converter, G2PText, G2PWord


class LexiconConverter(Converter, ABC):
    """Converter for languages where the writing system doubles as the
    pronunciation script (most alphabetical languages).

    Known words are looked up in a pronunciation dictionary loaded from
    *dict_path*.  Out-of-vocabulary words are handled by
    ``_infer_oov``, which subclasses implement with language-specific
    letter-to-sound rules.
    """

    def __init__(self, dict_path: str | None = None) -> None:
        super().__init__()
        self._dict: dict[str, list[list[str]]] = {}
        if dict_path is not None:
            from .dictionary import load_pronunciation_dict
            self._dict = load_pronunciation_dict(dict_path)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def infer_oov(self, token: str) -> list[list[str]]:
        """Infer phoneme sequences for an out-of-vocabulary token.

        Called when *token* is not found in the pronunciation dictionary.
        Returns one or more alternative phoneme sequences.
        """
        ...

    # ------------------------------------------------------------------
    # Converter interface
    # ------------------------------------------------------------------

    def convert(self, tokens: list[str]) -> list[G2PText]:
        result: list[G2PText] = []
        for token in tokens:
            pronunciations = self._dict.get(token)
            if pronunciations is not None:
                paths = [list(p) for p in pronunciations]
            else:
                paths = self.infer_oov(token)
            result.append(G2PText(text=token, words=[G2PWord(word=token, phones=paths)]))
        return result


class PronunciationScriptConverter(Converter, ABC):
    """Converter for writing systems that use a decoupled *pronunciation script*.

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

    def convert(self, tokens: list[str]) -> list[G2PText]:
        scripts_per_token = self.text_to_script(tokens)
        result: list[G2PText] = []
        for token, scripts in zip(tokens, scripts_per_token):
            words: list[G2PWord] = []
            seen_words: set[str] = set()
            for s in scripts:
                if s in seen_words:
                    continue
                seen_words.add(s)
                seen_paths: set[tuple[str, ...]] = set()
                phones: list[list[str]] = []
                for phonemes in self.script_to_phonemes(s):
                    key = tuple(phonemes)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        phones.append(list(phonemes))
                words.append(G2PWord(word=s, phones=phones))
            result.append(G2PText(text=token, words=words))
        return result
