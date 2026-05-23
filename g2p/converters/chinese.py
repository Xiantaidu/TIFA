"""Mandarin and Cantonese G2P converters — delegate to the cpp-pinyin engine.

Auto-discovered by ``g2p/converters/__init__.py``.
Both derive from ``PronunciationScriptDictionaryConverter``: hanzi → pinyin/jyutping
(text-to-script, via the cpp-pinyin engine) then pinyin/jyutping → phonemes
(script-to-phonemes, via dictionary lookup when *dict_path* is given).
"""


from pathlib import Path

from g2p.converters.cpp_pinyin import PinyinEngine
from g2p.converters.cpp_pinyin.constants import STYLE_NORMAL
from g2p.registry import converter

from .dictionary import PronunciationScriptDictionaryConverter

_CPP_PINYIN_DIR = Path(__file__).parent / "cpp_pinyin" / "dicts"


class _ChineseScriptConverter(PronunciationScriptDictionaryConverter):
    """Shared base for cpp-pinyin-backed Chinese converters.

    Subclasses are decorated with ``@converter`` to register language and id.
    """

    def __init__(
        self,
        dict_path: str,
        *,
        _bundled_dict: str,
    ) -> None:
        super().__init__(dict_path=dict_path)
        self._engine = PinyinEngine(_bundled_dict)

    @staticmethod
    def _is_hanzi(token: str) -> bool:
        if len(token) != 1:
            return False
        return 0x4E00 <= ord(token) <= 0x9FA5

    def claim(self, token: str) -> bool:
        return self._is_hanzi(token)

    def text_to_script(self, tokens: list[str]) -> list[list[str]]:
        simplified = self._engine.simplify(tokens)
        best = self._engine.query_raw(simplified, style=STYLE_NORMAL)
        result: list[list[str]] = []
        for ch, best_list in zip(simplified, best):
            primary = best_list[0]
            scripts = [primary]
            for reading in self._engine.readings(ch, style=STYLE_NORMAL):
                if reading != primary:
                    scripts.append(reading)
            result.append(scripts)
        return result


@converter(id="mandarin", language="zh,cmn")
class MandarinConverter(_ChineseScriptConverter):
    """Mandarin Chinese pinyin converter.

    Config examples::

        converters:
          - id: mandarin
          - id: passthrough
            language: eng

        # Pinyin-to-phoneme mapping
        converters:
          - id: mandarin
            kwargs:
              dict_path: /path/to/pinyin_phonemes.txt
    """

    def __init__(self, dict_path: str) -> None:
        super().__init__(
            dict_path=dict_path,
            _bundled_dict=str(_CPP_PINYIN_DIR / "mandarin"),
        )


@converter(id="cantonese", language="yue")
class CantoneseConverter(_ChineseScriptConverter):
    """Cantonese (Jyutping) converter.

    Config examples::

        converters:
          - id: cantonese
          - id: passthrough
            language: eng

        # Jyutping-to-phoneme mapping
        converters:
          - id: cantonese
            kwargs:
              dict_path: /path/to/jyutping_phonemes.txt
    """

    def __init__(self, dict_path: str) -> None:
        super().__init__(
            dict_path=dict_path,
            _bundled_dict=str(_CPP_PINYIN_DIR / "cantonese"),
        )
