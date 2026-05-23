import string

from g2p.registry import preprocessor

from .base import Preprocessor


@preprocessor(id="punctuation_filter")
class PunctuationFilter(Preprocessor):
    """Remove tokens that consist entirely of punctuation."""

    _punctuation_set = frozenset(string.punctuation)

    def process(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if not all(c in self._punctuation_set for c in t)]


@preprocessor(id="lowercase")
class LowercasePreprocessor(Preprocessor):
    """Lowercase all tokens."""

    def process(self, tokens: list[str]) -> list[str]:
        return [t.lower() for t in tokens]


@preprocessor(id="strip_whitespace")
class StripWhitespacePreprocessor(Preprocessor):
    """Strip leading and trailing whitespace from tokens, removing empty ones."""

    def process(self, tokens: list[str]) -> list[str]:
        return [s for t in tokens if (s := t.strip())]
