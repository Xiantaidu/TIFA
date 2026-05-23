from abc import ABC, abstractmethod
from dataclasses import dataclass

from g2p.registry import converter

from ..preprocessors.base import Preprocessor


class Converter(ABC):
    _language: str | None = None

    @property
    def language(self) -> str | None:
        return self._language

    @abstractmethod
    def claim(self, token: str) -> bool:
        ...

    # noinspection PyMethodMayBeStatic
    def preprocessors(self) -> list[Preprocessor]:
        return []

    @abstractmethod
    def convert(self, tokens: list[str]) -> list[list[str]]:
        ...


@dataclass
class _TokenState:
    text: str
    index: int
    phonemes: list[str] | None = None
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
    first converter that claims it; unconverted tokens raise G2PConversionError."""

    def __init__(self, modules: list[Converter]) -> None:
        self.modules = modules

    def claim(self, token: str) -> bool:
        return any(m.claim(token) for m in self.modules)

    def convert(self, tokens: list[str]) -> list[list[str]]:
        states = [_TokenState(text=t, index=i) for i, t in enumerate(tokens)]

        for module in self.modules:
            unconverted = [s for s in states if s.phonemes is None]
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

                phoneme_lists = module.convert(run_texts)

                for state, phonemes in zip(run_states, phoneme_lists):
                    state.phonemes = phonemes
                    state.assigned_converter = type(module).__name__
                i = j

        unconverted = [s for s in states if s.phonemes is None]
        if unconverted:
            raise G2PConversionError([s.text for s in unconverted])

        return [s.phonemes for s in states]
