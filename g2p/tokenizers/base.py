from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, tokens: list[str]) -> list[str]:
        ...
