from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..preprocessors.base import Preprocessor


@dataclass
class G2PWord:
    """A pronunciation-script word with its phoneme alternatives.

    ``word`` is the intermediate representation (pinyin, romaji, etc.).
    ``phones`` holds alternative phoneme sequences for this word.
    """

    word: str
    phones: list[list[str]] = field(default_factory=list)


@dataclass
class G2PText:
    """Three-tier G2P output for one text token.

    ``text`` is the original token from the tokenizer.
    ``words`` holds one or more pronunciation-script as word-level alternatives,
    each word alternative holding its own phoneme-level alternatives.
    *language* is set by the pipeline to the tag (e.g. ``"cmn"``) that
    caused this converter to be selected.  Converters with no language
    registration leave it ``None``.
    """

    text: str
    words: list[G2PWord] = field(default_factory=list)
    language: str | None = None


class Converter(ABC):
    language: tuple[str, ...] | None = None

    @abstractmethod
    def claim(self, token: str) -> bool:
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        return []

    @abstractmethod
    def convert(self, tokens: list[str]) -> list[G2PText]:
        ...


class G2PConversionError(Exception):
    def __init__(self, unconverted_tokens: list[str]) -> None:
        self.unconverted_tokens = unconverted_tokens
        super().__init__(
            f"The following tokens could not be converted "
            f"by any converter in the chain: {unconverted_tokens}"
        )


def resolve_language(
    language: tuple[str, ...] | None,
    language_set: set[str] | None,
) -> str | None:
    """Return the single language tag that matched, or *None*."""
    if language is None:
        return None
    if language_set is None:
        return language[0]
    for tag in language:
        if tag in language_set:
            return tag
    return None
