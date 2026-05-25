from dataclasses import dataclass

from .converters.base import (
    Converter,
    G2PConversionError,
    G2PText,
    resolve_language,
)
from .preprocessors.base import Preprocessor
from .tokenizers.base import Tokenizer


@dataclass
class _TokenState:
    text: str
    index: int
    result: G2PText | None = None


class G2PPipeline:
    def __init__(
        self,
        preprocessors: list[Preprocessor] | None = None,
        tokenizers: list[Tokenizer] | None = None,
        converters: list[Converter] | None = None,
    ) -> None:
        self._preprocessors = preprocessors or []
        self._tokenizers = tokenizers or []
        self._converters = converters or []

    def convert(
        self, text: str, *, languages: list[str] | None = None,
    ) -> list[G2PText]:
        language_set = set(languages) if languages else None
        active = [
            c for c in self._converters
            if language_set is None
            or c.language is None
            or any(ln in language_set for ln in c.language)
        ]
        if not active:
            raise ValueError("No converter matches the requested languages.")

        tokens = [text]
        for pp in self._preprocessors:
            tokens = pp.process(tokens)
        for tok in self._tokenizers:
            tokens = tok.tokenize(tokens)

        states = [_TokenState(text=t, index=i) for i, t in enumerate(tokens)]
        for converter in active:
            unconverted = [s for s in states if s.result is None]
            i = 0
            while i < len(unconverted):
                if not converter.claim(unconverted[i].text):
                    i += 1
                    continue
                j = i + 1
                while j < len(unconverted) and converter.claim(unconverted[j].text):
                    j += 1
                run_states = unconverted[i:j]
                run_texts = [s.text for s in run_states]
                for pp in converter.preprocessors():
                    run_texts = pp.process(run_texts)
                results = converter.convert(run_texts)
                resolved = resolve_language(converter.language, language_set)
                for state, result in zip(run_states, results):
                    result.language = resolved
                    state.result = result
                i = j

        unconverted = [s for s in states if s.result is None]
        if unconverted:
            raise G2PConversionError([s.text for s in unconverted])

        return [s.result for s in states]
