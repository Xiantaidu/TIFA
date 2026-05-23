"""Japanese kana G2P converter — kana → romaji → phonemes.

Auto-discovered by ``g2p/converters/__init__.py`` and registered as
``@converter(id="japanese_kana", language="ja")``.
"""

from __future__ import annotations

from g2p.registry import converter

from .dictionary import PronunciationScriptDictionaryConverter

# Kana → romaji mapping covering hiragana and katakana, including
# dakuten/handakuten variants and small-kana.
_KANA_TO_ROMAJI: dict[str, str] = {}

# Compact table: (hiragana, katakana, romaji)
_KANA_TABLE: list[tuple[str, str, str]] = [
    # ---- gojūon ----
    ("あ", "ア", "a"), ("い", "イ", "i"), ("う", "ウ", "u"), ("え", "エ", "e"), ("お", "オ", "o"),
    ("か", "カ", "ka"), ("き", "キ", "ki"), ("く", "ク", "ku"), ("け", "ケ", "ke"), ("こ", "コ", "ko"),
    ("さ", "サ", "sa"), ("し", "シ", "shi"), ("す", "ス", "su"), ("せ", "セ", "se"), ("そ", "ソ", "so"),
    ("た", "タ", "ta"), ("ち", "チ", "chi"), ("つ", "ツ", "tsu"), ("て", "テ", "te"), ("と", "ト", "to"),
    ("な", "ナ", "na"), ("に", "ニ", "ni"), ("ぬ", "ヌ", "nu"), ("ね", "ネ", "ne"), ("の", "ノ", "no"),
    ("は", "ハ", "ha"), ("ひ", "ヒ", "hi"), ("ふ", "フ", "fu"), ("へ", "ヘ", "he"), ("ほ", "ホ", "ho"),
    ("ま", "マ", "ma"), ("み", "ミ", "mi"), ("む", "ム", "mu"), ("め", "メ", "me"), ("も", "モ", "mo"),
    ("や", "ヤ", "ya"), ("ゆ", "ユ", "yu"), ("よ", "ヨ", "yo"),
    ("ら", "ラ", "ra"), ("り", "リ", "ri"), ("る", "ル", "ru"), ("れ", "レ", "re"), ("ろ", "ロ", "ro"),
    ("わ", "ワ", "wa"), ("ゐ", "ヰ", "wi"), ("ゑ", "ヱ", "we"), ("を", "ヲ", "wo"),
    ("ん", "ン", "n"),
    # ---- dakuten ----
    ("が", "ガ", "ga"), ("ぎ", "ギ", "gi"), ("ぐ", "グ", "gu"), ("げ", "ゲ", "ge"), ("ご", "ゴ", "go"),
    ("ざ", "ザ", "za"), ("じ", "ジ", "ji"), ("ず", "ズ", "zu"), ("ぜ", "ゼ", "ze"), ("ぞ", "ゾ", "zo"),
    ("だ", "ダ", "da"), ("ぢ", "ヂ", "ji"), ("づ", "ヅ", "zu"), ("で", "デ", "de"), ("ど", "ド", "do"),
    ("ば", "バ", "ba"), ("び", "ビ", "bi"), ("ぶ", "ブ", "bu"), ("べ", "ベ", "be"), ("ぼ", "ボ", "bo"),
    # ---- handakuten ----
    ("ぱ", "パ", "pa"), ("ぴ", "ピ", "pi"), ("ぷ", "プ", "pu"), ("ぺ", "ペ", "pe"), ("ぽ", "ポ", "po"),
    # ---- small kana ----
    ("ゃ", "ャ", "ya"), ("ゅ", "ュ", "yu"), ("ょ", "ョ", "yo"),
    ("ぁ", "ァ", "a"), ("ぃ", "ィ", "i"), ("ぅ", "ゥ", "u"), ("ぇ", "ェ", "e"), ("ぉ", "ォ", "o"),
    ("っ", "ッ", "q"),
    # ---- katakana-only ----
    ("ヴ", "ヴ", "vu"),
    ("ー", "ー", ":"),
]

for _h, _k, _r in _KANA_TABLE:
    _KANA_TO_ROMAJI[_h] = _r
    _KANA_TO_ROMAJI[_k] = _r

_KANA_CHARS = frozenset(_KANA_TO_ROMAJI)


@converter(id="japanese_kana", language="ja")
class JapaneseKanaConverter(PronunciationScriptDictionaryConverter):
    """Japanese kana-to-phoneme converter.

    Two-phase: each kana character → romaji (text-to-script), then
    romaji → phonemes (script-to-phonemes, via dictionary when
    *dict_path* is given).

    Handles both hiragana and katakana.  Each kana maps to its romaji
    individually; yōon digraphs (e.g. きゃ → kya) are composed in
    preprocessing before tokenization, or handled through the dictionary.
    Small tsu (っ/ッ) maps to ``q`` so a dictionary can resolve gemination
    (e.g. ``qka`` → ``["k", "k", "a"]``).
    """

    def __init__(self, dict_path: str | None = None) -> None:
        super().__init__(dict_path=dict_path)

    def claim(self, token: str) -> bool:
        if len(token) != 1:
            return False
        return token in _KANA_CHARS

    def text_to_script(self, tokens: list[str]) -> list[list[str]]:
        return [[_KANA_TO_ROMAJI.get(t, t)] for t in tokens]
