from g2p.registry import converter

from .base import Converter, PronunciationGroup


@converter(id="passthrough", language=None)
class PassthroughConverter(Converter):
    """Catch-all converter that returns each token as its own phoneme.
    Typically placed last in a chain as a fallback for unconverted tokens."""

    def claim(self, token: str) -> bool:
        return True

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        return [PronunciationGroup(paths=[[t]]) for t in tokens]


@converter(id="characters", language=None)
class CharPhonemeConverter(Converter):
    """One-to-one character-to-phoneme mapping.
    Each character in a token is mapped to one or more phonemes."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def claim(self, token: str) -> bool:
        return all(c in self._mapping for c in token)

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        result: list[PronunciationGroup] = []
        for token in tokens:
            phonemes: list[str] = []
            for c in token:
                phonemes.extend(self._mapping[c])
            result.append(PronunciationGroup(paths=[phonemes]))
        return result
