from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from g2p.registry import converter

from ..preprocessors.base import Preprocessor


@dataclass
class PronunciationGroup:
    """A collection of alternative phoneme sequences for one token.

    Each path in ``paths`` represents one possible pronunciation as a list
    of phoneme strings.  Converters that only produce a single pronunciation
    per token still wrap it in a one-element ``paths`` list.
    """

    paths: list[list[str]] = field(default_factory=list)


class Converter(ABC):
    _language: tuple[str, ...] | None = None

    @property
    def language(self) -> tuple[str, ...] | None:
        return self._language

    @abstractmethod
    def claim(self, token: str) -> bool:
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        return []

    @abstractmethod
    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        ...


@dataclass
class _TokenState:
    text: str
    index: int
    pronunciation: PronunciationGroup | None = None
    assigned_converter: str | None = None


class G2PConversionError(Exception):
    def __init__(self, unconverted_tokens: list[str]) -> None:
        self.unconverted_tokens = unconverted_tokens
        super().__init__(
            f"The following tokens could not be converted "
            f"by any converter in the chain: {unconverted_tokens}"
        )


@converter(id="chain", language=None)
class ChainedConverter(Converter):
    """Chain multiple converters in priority order. Each token is handled by the
    first converter that claims it; unconverted tokens raise G2PConversionError.

    If *language* is given (as a comma-separated string), the chain declares
    that language so it can be filtered by ``G2PPipeline`` alongside
    single-language converters.
    """

    def __init__(
        self,
        modules: list[Converter],
        language: str | None = None,
    ) -> None:
        self.modules = modules
        if language is not None:
            from g2p.registry import _parse_language
            self._language = _parse_language(language)

    def claim(self, token: str) -> bool:
        return any(m.claim(token) for m in self.modules)

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        states = [_TokenState(text=t, index=i) for i, t in enumerate(tokens)]

        for module in self.modules:
            unconverted = [s for s in states if s.pronunciation is None]
            i = 0
            while i < len(unconverted):
                if not module.claim(unconverted[i].text):
                    i += 1
                    continue
                j = i + 1
                while j < len(unconverted) and module.claim(unconverted[j].text):
                    j += 1
                run_states = unconverted[i:j]

                run_texts = [s.text for s in run_states]
                for pp in module.preprocessors():
                    run_texts = pp.process(run_texts)

                results = module.convert(run_texts)

                for state, result in zip(run_states, results):
                    state.pronunciation = result
                    state.assigned_converter = type(module).__name__
                i = j

        unconverted = [s for s in states if s.pronunciation is None]
        if unconverted:
            raise G2PConversionError([s.text for s in unconverted])

        return [s.pronunciation for s in states]
