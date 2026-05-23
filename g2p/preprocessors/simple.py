import string

from g2p.registry import preprocessor

from .base import Preprocessor


@preprocessor(id="punctuation_filter")
class PunctuationFilter(Preprocessor):
    """Split tokens on punctuation and discard the punctuation characters."""

    _punctuation_set = frozenset(
        string.punctuation
        + "，。；：“”‘’（）【】《》…—～、·"
        + "！？"
    )

    def process(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for t in tokens:
            part: list[str] = []
            for c in t:
                if c in self._punctuation_set:
                    if part:
                        result.append("".join(part))
                        part = []
                else:
                    part.append(c)
            if part:
                result.append("".join(part))
        return result


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
