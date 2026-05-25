from g2p.registry import tokenizer
from .base import Tokenizer


@tokenizer(id="whitespace")
class WhitespaceTokenizer(Tokenizer):
    """Split tokens on Unicode whitespace boundaries."""

    def tokenize(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for token in tokens:
            result.extend(token.split())
        return result


@tokenizer(id="character")
class CharacterTokenizer(Tokenizer):
    """Split each token into individual characters."""

    def tokenize(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for token in tokens:
            result.extend(list(token))
        return result


@tokenizer(id="identity")
class IdentityTokenizer(Tokenizer):
    """Return tokens unchanged."""

    def tokenize(self, tokens: list[str]) -> list[str]:
        return list(tokens)
