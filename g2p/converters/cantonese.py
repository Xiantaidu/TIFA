"""Cantonese G2P wrapper — delegates to the faithful cpp-pinyin engine.

Auto-discovered by ``g2p/converters/__init__.py`` and registered as
``@converter(id="cantonese", language="yue")``.
"""

from __future__ import annotations

from pathlib import Path

from g2p.converters.base import Converter, PronunciationGroup
from g2p.converters.cpp_pinyin import PinyinEngine
from g2p.converters.cpp_pinyin.constants import STYLE_NORMAL
from g2p.registry import converter


@converter(id="cantonese", language="yue")
class CantoneseConverter(Converter):
    """Cantonese (Jyutping) converter.

    Uses the bundled cpp-pinyin dictionaries by default.

    Config examples::

        # Bundled dictionaries (no kwargs needed)
        converters:
          - id: cantonese
          - id: passthrough

        # Custom dictionaries
        converters:
          - id: cantonese
            kwargs:
              dict_dir: /path/to/dicts
    """

    _BUNDLED_DICT = str(Path(__file__).parent / "cpp_pinyin" / "dicts" / "cantonese")

    def __init__(self, dict_dir: str | None = None) -> None:
        self._engine = PinyinEngine(dict_dir or self._BUNDLED_DICT)

    @staticmethod
    def _is_hanzi(token: str) -> bool:
        if len(token) != 1:
            return False
        return 0x4E00 <= ord(token) <= 0x9FA5

    def claim(self, token: str) -> bool:
        return self._is_hanzi(token)

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        simplified = self._engine.simplify(tokens)
        best = self._engine.query_raw(simplified, style=STYLE_NORMAL)
        result: list[PronunciationGroup] = []
        for ch, best_list in zip(simplified, best):
            seen: set[tuple[str, ...]] = set()
            paths: list[list[str]] = []
            # best_list e.g. ["nei"] — phrase-disambiguated best path
            key = tuple(best_list)
            seen.add(key)
            paths.append(best_list)
            # Alternative readings from dictionary
            for reading in self._engine.readings(ch, style=STYLE_NORMAL):
                p = [reading]
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    paths.append(p)
            result.append(PronunciationGroup(paths=paths))
        return result
