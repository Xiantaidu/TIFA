from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, tokens: list[str]) -> list[str]:
        ...


class ChainedTokenizer(Tokenizer):
    def __init__(self, tokenizers: list[Tokenizer]) -> None:
        self._tokenizers = tokenizers

    def tokenize(self, tokens: list[str]) -> list[str]:
        for tok in self._tokenizers:
            tokens = tok.tokenize(tokens)
        return tokens
