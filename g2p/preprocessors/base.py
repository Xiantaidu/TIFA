from abc import ABC, abstractmethod


class Preprocessor(ABC):
    @abstractmethod
    def process(self, tokens: list[str]) -> list[str]:
        ...


class ChainedPreprocessor(Preprocessor):
    def __init__(self, preprocessors: list[Preprocessor]) -> None:
        self._preprocessors = preprocessors

    def process(self, tokens: list[str]) -> list[str]:
        for pp in self._preprocessors:
            tokens = pp.process(tokens)
        return tokens
