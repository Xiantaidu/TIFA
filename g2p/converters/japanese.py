"""Japanese kana G2P converter  --  kana -> romaji -> phonemes.
"""

from g2p.registry import converter
from .dictionary import PronunciationScriptDictionaryConverter

# Small kana used in yōon digraphs and other digraphs.
_SMALL_KANA = frozenset("ゃゅょャュョぁぃぅぇぉァィゥェォ")

# Hiragana-only romaji table, derived from cpp-kana's kanaToRomajiMap.
# Katakana is converted to hiragana before lookup.
_KANA_TO_ROMAJI: dict[str, str] = {
    # ---- sokuon ----
    "っ": "cl",
    # ---- gojūon ----
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wi", "ゑ": "we",
    "ん": "n",
    # ---- dakuten ----
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    # ---- handakuten ----
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    # ---- yōon / digraphs ----
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo", "きぇ": "kye",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo", "ぎぇ": "gye",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "じゃ": "ja",  "じゅ": "ju",  "じょ": "jo",  "じぇ": "je",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo", "にぇ": "nye",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo", "ひぇ": "hye",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo", "びぇ": "bye",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo", "ぴぇ": "pye",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo", "みぇ": "mye",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo", "りぇ": "rye",
    # ---- vowel-extension digraphs ----
    "いぇ": "ye",
    "うぁ": "wa", "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "くぁ": "kwa", "くぃ": "kwi", "くぇ": "kwe", "くぉ": "kwo",
    "ぐぁ": "gwa", "ぐぃ": "gwi", "ぐぇ": "gwe", "ぐぉ": "gwo",
    "すぁ": "swa", "すぃ": "swi", "すぇ": "swe", "すぉ": "swo",
    "ずぁ": "zwa", "ずぃ": "zwi", "ずぇ": "zwe", "ずぉ": "zwo",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
    "てぃ": "ti", "てゅ": "tyu",
    "でぃ": "di", "でゅ": "dyu",
    "とぅ": "tu",
    "どぅ": "du",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "ぶぁ": "bwa", "ぶぃ": "bwi", "ぶぇ": "bwe", "ぶぉ": "bwo",
    "ぷぁ": "pwa", "ぷぃ": "pwi", "ぷぇ": "pwe", "ぷぉ": "pwo",
    "ぬぁ": "nwa", "ぬぃ": "nwi", "ぬぇ": "nwe", "ぬぉ": "nwo",
    "むぁ": "mwa", "むぃ": "mwi", "むぇ": "mwe", "むぉ": "mwo",
    "るぁ": "rwa", "るぃ": "rwi", "るぇ": "rwe", "るぉ": "rwo",
    # ---- ヴ variants (hiragana form ゔ) ----
    "ゔ": "vu",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    # ---- special ----
    "を": "o",
}

_HIRAGANA_START = 0x3041
_KATAKANA_START = 0x30A1
_KANA_SPAN = 0x5E


def _kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana by shifting the Unicode range."""
    result: list[str] = []
    for ch in text:
        cp = ord(ch)
        if _KATAKANA_START <= cp < _KATAKANA_START + _KANA_SPAN:
            result.append(chr(cp - _KATAKANA_START + _HIRAGANA_START))
        else:
            result.append(ch)
    return "".join(result)


def _is_kana(ch: str) -> bool:
    cp = ord(ch)
    return (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF)


_CONSONANT_LEADING = frozenset(
    "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
)


def _apply_sokuon(romaji_list: list[str]) -> list[str]:
    """Resolve gemination: replace 'cl' with the leading consonant of the
    next non-empty romaji token.  Empty strings (placeholders for skipped
    kana) are passed through unchanged.  Returns a list of the same length."""
    result: list[str] = []
    i = 0
    while i < len(romaji_list):
        r = romaji_list[i]
        if r == "cl":
            # Find next non-placeholder romaji
            j = i + 1
            while j < len(romaji_list) and romaji_list[j] == "":
                j += 1
            if j < len(romaji_list):
                nxt = romaji_list[j]
                if nxt[0] in _CONSONANT_LEADING:
                    result.append(nxt[0])
                    i += 1
                    continue
        result.append(r)
        i += 1
    return result


@converter(id="japanese-kana", language="ja,jpn")
class JapaneseKanaConverter(PronunciationScriptDictionaryConverter):
    """Japanese kana-to-phoneme converter.

    Two-phase: kana -> romaji (text-to-script), then
    romaji -> phonemes (script-to-phonemes via dictionary).

    Parameters mirror cpp-kana:
        *dict_path*: romaji-to-phoneme dictionary.
        *double_written_sokuon*: enable gemination resolution
          (``cl`` + consonant -> consonant gemination).
    """

    def __init__(
        self,
        dict_path: str,
        *,
        double_written_sokuon: bool = False,
    ) -> None:
        super().__init__(dict_path=dict_path)
        self._double_written_sokuon = double_written_sokuon

    def claim(self, token: str) -> bool:
        if not token:
            return False
        # Single kana
        if len(token) == 1:
            return _is_kana(token)
        # Kana digraph: two chars, first is kana, second is small kana
        if len(token) == 2:
            return _is_kana(token[0]) and token[1] in _SMALL_KANA
        return False

    def script_to_phonemes(self, script: str) -> list[list[str]]:
        if script == "cl":
            return [["cl"]]
        if script == "":
            return [[]]
        if script in self._script_dict:
            return [list(p) for p in self._script_dict[script]]
        # Single consonants from gemination pass through directly
        if len(script) == 1 and script in _CONSONANT_LEADING:
            return [[script]]
        raise KeyError(
            f"Script token {script!r} not found in script-to-phoneme dict."
        )

    def text_to_script(self, tokens: list[str]) -> list[list[str]]:
        # Convert katakana to hiragana for unified lookup
        hiragana_tokens = [_kata_to_hira(t) for t in tokens]

        # Kana -> romaji; ー (long vowel) and ゜ (handakuten) produce empty
        romaji_list: list[str] = []
        for t in hiragana_tokens:
            if t in ("ー", "゜"):
                romaji_list.append("")
                continue
            r = _KANA_TO_ROMAJI.get(t)
            if r is not None:
                romaji_list.append(r)
            else:
                romaji_list.append(t)  # passthrough  --  shouldn't happen if claim() is correct

        if self._double_written_sokuon:
            romaji_list = _apply_sokuon(romaji_list)

        return [[r] for r in romaji_list]
