from g2p.registry import converter
from .base import Converter, G2PGroup, G2PWord, G2PReading


@converter(id="passthrough", language=None)
class PassthroughConverter(Converter):
    """Catch-all converter that returns each token as its own phoneme.
    Typically placed last in a chain as a fallback for unconverted tokens."""

    def claim(self, token: str) -> bool:
        return True

    def convert(self, words: list[str]) -> list[G2PWord]:
        return [G2PWord(text=t, readings=[G2PReading(paths=[
            [G2PGroup(script=t, phonemes=[t])],
        ])]) for t in words]


@converter(id="characters", language=None)
class CharPhonemeConverter(Converter):
    """One-to-one character-to-phoneme mapping.
    Each character in a token is mapped to one or more phonemes."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def claim(self, token: str) -> bool:
        return all(c in self._mapping for c in token)

    def convert(self, words: list[str]) -> list[G2PWord]:
        result: list[G2PWord] = []
        for token in words:
            phonemes: list[str] = []
            for c in token:
                phonemes.extend(self._mapping[c])
            path = [G2PGroup(script=token, phonemes=phonemes)] if phonemes else []
            result.append(G2PWord(text=token, readings=[G2PReading(paths=[path])]))
        return result
