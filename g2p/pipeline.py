from .converters.base import ChainedConverter, Converter, PronunciationGroup
from .preprocessors.base import Preprocessor
from .tokenizers.base import Tokenizer


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

    def convert(self, text: str, *, languages: list[str] | None = None) -> list[PronunciationGroup]:
        language_set = set(languages) if languages else None
        active = [
            c for c in self._converters
            if language_set is None or c.language is None or c.language in language_set
        ]
        if not active:
            raise ValueError("No converter matches the requested languages.")

        tokens = [text]
        for pp in self._preprocessors:
            tokens = pp.process(tokens)
        for tok in self._tokenizers:
            tokens = tok.tokenize(tokens)

        converter = active[0] if len(active) == 1 else ChainedConverter(modules=active)
        return converter.convert(tokens)
