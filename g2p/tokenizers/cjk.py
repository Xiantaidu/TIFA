from g2p.registry import tokenizer

from .base import Tokenizer


def _is_cjk(c: str) -> bool:
    cp = ord(c)
    return (
        0x2E80 <= cp <= 0x2EFF  # CJK Radicals Supplement
        or 0x2F00 <= cp <= 0x2FDF  # Kangxi Radicals
        or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x31F0 <= cp <= 0x31FF  # Katakana Phonetic Extensions
        or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth and Fullwidth Forms
    )


def _is_kana(c: str) -> bool:
    cp = ord(c)
    return 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF


_SMALL_KANA = frozenset(
    "ゃゅょャュョぁぃぅぇぉァィゥェォ"
)


def _is_small_kana(c: str) -> bool:
    return c in _SMALL_KANA


@tokenizer(id="cjk")
class CJKTokenizer(Tokenizer):
    """Split CJK characters individually while grouping non-CJK characters into runs.
    ``你好hello世界`` → ``[你, 好, hello, 世界]``.

    Kana digraphs (kana + small kana) are kept together:
    ``きゃ`` → ``[きゃ]``."""

    def tokenize(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        for token in tokens:
            chars = list(token)
            i = 0
            while i < len(chars):
                c = chars[i]
                if _is_kana(c) and i + 1 < len(chars) and _is_small_kana(chars[i + 1]):
                    # kana digraph — group with following small kana
                    result.append(c + chars[i + 1])
                    i += 2
                elif _is_cjk(c):
                    result.append(c)
                    i += 1
                else:
                    buf: list[str] = []
                    while i < len(chars) and not _is_cjk(chars[i]):
                        buf.append(chars[i])
                        i += 1
                    result.append("".join(buf))
        return result
