"""Convert cpp-pinyin dictionary files to FA2026 G2P TSV format.

Reads from D:\projects\cpp-pinyin\res\dict\{mandarin,cantonese}\
Writes to D:\python\FA2026\dictionaries\{mandarin,cantonese}\

cpp-pinyin word.txt format:  汉字:拼音1,拼音2,...
  → FA2026 TSV: 汉字\t拼音1\n汉字\t拼音2...

cpp-pinyin phrases_dict.txt: 词组:拼音1,拼音2,...
  → FA2026 TSV: 词组\t拼音1 拼音2...

cpp-pinyin trans_word.txt:   繁体字:简体字
  → Used to add traditional-char entries pointing to simplified-char's phonemes
"""

import re
import sys
from pathlib import Path

SRC = Path(r"D:\projects\cpp-pinyin\res\dict")
DST = Path(r"D:\python\FA2026\dictionaries")

# Unicode tone-mark → (base_char, tone_number)
# Covers all tone marks used in cpp-pinyin dictionaries
TONE_MAP: dict[int, tuple[str, str]] = {
    # a with tones
    0x0101: ("a", "1"), 0x00E1: ("a", "2"), 0x01CE: ("a", "3"), 0x00E0: ("a", "4"),
    # e with tones
    0x0113: ("e", "1"), 0x00E9: ("e", "2"), 0x011B: ("e", "3"), 0x00E8: ("e", "4"),
    # i with tones
    0x012B: ("i", "1"), 0x00ED: ("i", "2"), 0x01D0: ("i", "3"), 0x00EC: ("i", "4"),
    # o with tones
    0x014D: ("o", "1"), 0x00F3: ("o", "2"), 0x01D2: ("o", "3"), 0x00F2: ("o", "4"),
    # u with tones
    0x016B: ("u", "1"), 0x00FA: ("u", "2"), 0x01D4: ("u", "3"), 0x00F9: ("u", "4"),
    # v (for ü) with tones  — ǖ ǘ ǚ ǜ
    0x01D6: ("v", "1"), 0x01D8: ("v", "2"), 0x01DA: ("v", "3"), 0x01DC: ("v", "4"),
    # plain ü (no tone) → v with neutral tone
    0x00FC: ("v", "5"),
    # syllabic n, m
    0x0144: ("n", "2"), 0x0148: ("n", "3"), 0x01F9: ("n", "4"),
    0x1E3F: ("m", "2"),
}


def tone_to_tone3(pinyin: str) -> str:
    """Convert a pinyin syllable to TONE3 (number) form.

    Handles two input styles:
      1. Tone marks:  zhōng, nǚ, ér, bù, le (neutral)
      2. TONE3 nums:  zung1, tau4, aa1, liu5

    Examples:
        zhōng → zhong1      nǚ → nv3
        le → le5             tau4 → tau4
        aa1 → aa1            zung1 → zung1
    """
    # Already in TONE3 format (ends with digit 1-9)
    if pinyin and pinyin[-1].isdigit():
        # Also convert any ü to v for consistency
        return pinyin.replace("ü", "v").replace("Ü", "V")

    result: list[str] = []
    tone: str | None = None
    for ch in pinyin:
        cp = ord(ch)
        if cp in TONE_MAP:
            base, t = TONE_MAP[cp]
            result.append(base)
            if t is not None:
                tone = t
        else:
            result.append(ch)
    if tone is None:
        tone = "5"
    return "".join(result) + tone


def is_cjk_char(ch: str) -> bool:
    """Check if a character is a CJK Unified Ideograph."""
    cp = ord(ch)
    return (0x3400 <= cp <= 0x4DBF) or (0x4E00 <= cp <= 0x9FFF) or (0xF900 <= cp <= 0xFAFF)


def load_trans(path: Path) -> dict[str, str]:
    """Load trans_word.txt → dict[traditional] = simplified."""
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        trad, simp = parts
        mapping[trad] = simp
    return mapping


def convert_word(in_path: Path, out_path: Path, trans: dict[str, str] | None = None) -> None:
    """Convert word.txt to FA2026 TSV format.

    cpp-pinyin: 汉字:拼音1,拼音2,...
    FA2026:     汉字\t拼音1\n汉字\t拼音2...
    """
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        char, pron_str = parts
        # Skip non-CJK entries (e.g., π:pài)
        if len(char) != 1 or not is_cjk_char(char):
            continue

        pronunciations = [p.strip() for p in pron_str.split(",") if p.strip()]
        for pron in pronunciations:
            tone3 = tone_to_tone3(pron)
            key = (char, tone3)
            if key not in seen:
                seen.add(key)
                lines.append(f"{char}\t{tone3}")

        # If trans map provided, add entries for traditional forms
        if trans is not None:
            for trad, simp in trans.items():
                if simp == char:
                    for pron in pronunciations:
                        tone3 = tone_to_tone3(pron)
                        key = (trad, tone3)
                        if key not in seen:
                            seen.add(key)
                            lines.append(f"{trad}\t{tone3}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {len(lines)} entries → {out_path}")


def convert_phrases(in_path: Path, out_path: Path) -> None:
    """Convert phrases_dict.txt to FA2026 TSV format.

    cpp-pinyin: 词组:拼音1,拼音2,...
    FA2026:     词组\t拼音1 拼音2 ...
    """
    lines: list[str] = []
    seen: set[str] = set()

    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        phrase, pron_str = parts
        if len(phrase) < 2:
            continue

        pronunciations = [p.strip() for p in pron_str.split(",") if p.strip()]
        tone3_list = [tone_to_tone3(p) for p in pronunciations]
        tsv_line = f"{phrase}\t{' '.join(tone3_list)}"
        if tsv_line not in seen:
            seen.add(tsv_line)
            lines.append(tsv_line)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {len(lines)} entries → {out_path}")


def main() -> None:
    for lang in ("mandarin", "cantonese"):
        print(f"\n=== {lang} ===")
        src_dir = SRC / lang
        dst_dir = DST / lang
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Load traditional→simplified mapping
        trans_path = src_dir / "trans_word.txt"
        trans = load_trans(trans_path) if trans_path.exists() else {}
        print(f"  Loaded {len(trans)} traditional→simplified mappings")

        # Convert word.txt
        convert_word(src_dir / "word.txt", dst_dir / "word.txt", trans)

        # Convert phrases_dict.txt
        convert_phrases(src_dir / "phrases_dict.txt", dst_dir / "phrases.txt")


if __name__ == "__main__":
    main()
