# G2P (Grapheme-to-Phoneme)

A multilingual grapheme-to-phoneme pipeline that converts raw text into phoneme sequences.

## Pipeline

```mermaid
flowchart LR
    Text["text<br/>(str)"] --> PP[preprocess]
    PP --> Tok[tokenize]
    Tok --> Conv[convert]
    Conv --> Ph["PronunciationGroup sequences<br/>(list[PronunciationGroup])"]
```

| Stage      | Input                | Output                       | Purpose                                                   |
|------------|----------------------|------------------------------|-----------------------------------------------------------|
| Preprocess | raw text as `[text]` | modified `[text]`            | Clean text before tokenization (lowercase, strip, filter) |
| Tokenize   | `list[str]`          | `list[str]`                  | Split text into tokens (words, characters, etc.)          |
| Convert    | `list[str]`          | `list[PronunciationGroup]`   | Convert tokens to phoneme sequences with language tags    |

Each stage runs its components sequentially — each feeds its output to the next.

### PronunciationGroup

```python
@dataclass
class PronunciationGroup:
    paths: list[list[str]]     # alternative phoneme sequences for one token
    language: str | None = None  # e.g. "cmn", "yue" — set by the pipeline
```

Pipeline output is `list[PronunciationGroup]`, one per token. Downstream code decides how to use the language tag (phoneme prefixing, filtering, etc.).

## Architecture

### Registry

Components are registered via decorators and looked up by ID at construction time.

```python
from g2p.registry import tokenizer, preprocessor, converter

@tokenizer(id="my-tokenizer")
class MyTokenizer(Tokenizer): ...

@preprocessor(id="my-preprocessor")
class MyPreprocessor(Preprocessor): ...

@converter(id="my-converter", language="eng")
class MyConverter(Converter): ...
```

`language` is a comma-separated string (e.g. `"zh,cmn"`) stored as `tuple[str, ...]` on the class.

### Auto-discovery

Dropping a `.py` file into `g2p/tokenizers/`, `g2p/preprocessors/`, or `g2p/converters/` automatically imports it and fires the decorator.

```
g2p/
├── registry.py         # @tokenizer, @preprocessor, @converter + lookup + parse_language()
├── pipeline.py         # G2PPipeline: preprocess→tokenize→convert loop with language stamping
├── api.py              # build_*_from_config(root_path=)
├── tokenizers/
│   ├── base.py         # Tokenizer ABC
│   ├── simple.py       # Whitespace, Character, Identity
│   └── cjk.py          # CJKTokenizer (also groups kana digraphs)
├── preprocessors/
│   ├── base.py         # Preprocessor ABC
│   └── simple.py       # FilterPunctuation, LowercasePreprocessor, StripWhitespacePreprocessor, RemoveAccentsPreprocessor
└── converters/
    ├── base.py         # Converter ABC, PronunciationGroup, G2PConversionError, resolve_language()
    ├── paradigm.py     # LexiconConverter, PronunciationScriptConverter
    ├── dictionary.py   # load_pronunciation_dict(), DictionaryConverter, PronunciationScriptDictionaryConverter
    ├── chinese.py      # MandarinConverter (id=chinese-pinyin), CantoneseConverter (id=cantonese-jyutping)
    ├── japanese.py     # JapaneseKanaConverter (id=japanese-kana) — cpp-kana-aligned
    ├── lstm.py         # LSTMConverter (id=lstm) — ONNX encoder-decoder OOV
    ├── simple.py       # PassthroughConverter, CharPhonemeConverter
    └── cpp_pinyin/     # PinyinEngine + dicts (mandarin, cantonese)
```

### Tokenizer

Splits tokens into smaller tokens. The first tokenizer receives `[raw_text]`.

**Built-in:**

| ID           | Class                 | Behavior                                                         |
|--------------|-----------------------|------------------------------------------------------------------|
| `whitespace` | `WhitespaceTokenizer` | Split on Unicode whitespace boundaries                           |
| `cjk`        | `CJKTokenizer`        | Split CJK chars individually, group non-CJK runs. Kana digraphs (kana + small kana) kept as single tokens |
| `character`  | `CharacterTokenizer`  | Split each token into individual characters                      |
| `identity`   | `IdentityTokenizer`   | Return tokens unchanged                                          |

### Preprocessor

Transforms a token sequence. Applied before tokenization — receives `[raw_text]`.

**Built-in:**

| ID                  | Class                         | Behavior                                             |
|---------------------|-------------------------------|------------------------------------------------------|
| `filter-punctuation` | `FilterPunctuation`           | Split tokens on punctuation and discard punctuation chars |
| `lowercase`         | `LowercasePreprocessor`       | Lowercase all tokens                                 |
| `strip-whitespace`  | `StripWhitespacePreprocessor` | Strip whitespace, remove empty tokens                |
| `remove-accents`    | `RemoveAccentsPreprocessor`   | Decompose accented chars and strip combining marks   |

### Converter

Converts tokens to phoneme sequences. Each converter implements:

- `claim(token) → bool` — whether this converter handles the given token
- `convert(tokens) → list[PronunciationGroup]` — convert tokens to phoneme groups
- `preprocessors() → list[Preprocessor]` (optional) — private preprocessors for claimed tokens

`Converter.language` is a class attribute `tuple[str, ...] | None`. The pipeline resolves which specific tag matched and stamps it on each `PronunciationGroup.language`. Converters with `language=None` leave the field `None`.

**Built-in converters:**

| ID                   | Class                  | Behavior                                                                     |
|----------------------|------------------------|------------------------------------------------------------------------------|
| `dictionary`         | `DictionaryConverter`  | Pronunciation dictionary lookup from a tab-separated file                    |
| `passthrough`        | `PassthroughConverter` | Catch-all: returns each token as its own phoneme                             |
| `characters`         | `CharPhonemeConverter` | One-to-one character-to-phoneme mapping                                      |
| `chinese-pinyin`     | `MandarinConverter`    | hanzi → pinyin (cpp-pinyin engine) → phonemes (dict)                         |
| `cantonese-jyutping` | `CantoneseConverter`   | hanzi → jyutping (cpp-pinyin engine) → phonemes (dict)                       |
| `japanese-kana`      | `JapaneseKanaConverter`| kana → romaji (cpp-kana-aligned table) → phonemes (dict). Handles yōon digraphs, gemination |
| `lstm`               | `LSTMConverter`        | LexiconConverter with ONNX encoder-decoder inference for OOV words           |

### Japanese Kana converter

Extends `PronunciationScriptDictionaryConverter`. Kana tokens (from CJKTokenizer, including digraphs like きゃ) are mapped to romaji via a table matching cpp-kana, then looked up in the dictionary.

Key behaviors:
- っ → `"cl"`, を → `"o"`; ー and ゜ produce empty phonemes
- Katakana auto-converted to hiragana before lookup
- `claim()` accepts single kana and 2-char digraphs (kana + small kana)
- `double_written_sokuon: bool = False` — gemination: `cl` + consonant → duplicate consonant (e.g. っか → k, ka)
- `script_to_phonemes` handles `cl` → `[["cl"]]`, empty string → `[[]]`, and single consonants from gemination → `[[consonant]]`

### LSTM Converter

Extends `LexiconConverter`. Dictionary lookup first, ONNX encoder-decoder inference for out-of-vocabulary words.

Parameters:
- `dict_path: str | None` — optional pronunciation dictionary
- `model_path: str` — directory with `encoder.onnx`, `decoder.onnx`, `char.json`, `phonemes.json`

`claim()` returns True if the token is in the dictionary OR all characters are in the model's char vocabulary. ONNX sessions are lazily loaded on first inference.

### Paradigms

Paradigm base classes live in `paradigm.py` and `dictionary.py`. They are **not** registered — subclasses add the `@converter` decorator.

**`PronunciationScriptConverter`** — for writing systems that use a decoupled pronunciation script (pinyin, jyutping, romaji). Two-phase:

1. `text_to_script(tokens) → list[list[str]]` — text → script tokens (with alternatives)
2. `script_to_phonemes(script) → list[list[str]]` — script token → phoneme sequences

Both methods are abstract. `convert()` orchestrates them and deduplicates paths.

**`PronunciationScriptDictionaryConverter`** — extends `PronunciationScriptConverter`. Fills in `script_to_phonemes` via `load_pronunciation_dict()`. Subclasses implement `text_to_script` and pass a required `dict_path` to the constructor.

**`LexiconConverter`** — for alphabetical languages where text IS the pronunciation script. Has a pronunciation dictionary for known words and an abstract `infer_oov(token)` method for out-of-vocabulary inference. `dict_path` is optional (pure inference is valid).

### Language resolution

When the pipeline runs with `languages=["cmn"]`:

1. Converters are filtered: a converter runs if `language is None` or any of its tags are in the set.
2. For each converter, `resolve_language(converter.language, language_set)` picks the single matching tag (or first tag when no filter is set).
3. The tag is stamped on each output `PronunciationGroup.language`.

### Convert algorithm

The pipeline processes tokens in priority order through each active converter:

1. Find the next contiguous run of unconverted tokens where `converter.claim()` is true.
2. Apply the converter's private preprocessors to the run.
3. Call `converter.convert()` on the run.
4. Stamp the resolved language on each result. Assign pronuncations back.
5. Repeat until the converter has no more claimed runs. Move to the next converter.

Any tokens still unconverted after all converters raise `G2PConversionError`.

### DictionaryConverter format

Tab-separated file: `<word>\t<ph1> <ph2> ...`

- Multiple pronunciations via duplicate entries or `(N)` / ` (N)` suffixes: `word`, `word(1)`, `word (2)`
- Case-insensitive
- All pronunciation variants are preserved in `paths`

```
hello	hh ax l ow
hello(1)	hh eh l ow
world	w er l d
```

### Shared dictionary loader

`load_pronunciation_dict(path)` in `dictionary.py` loads a tab-separated file and returns `dict[str, list[list[str]]]`. Used by `DictionaryConverter`, `PronunciationScriptDictionaryConverter`, `LSTMConverter`, and any custom converter that needs script-to-phoneme lookup. No `(N)`-variant handling — that's `DictionaryConverter`-specific.

## Configuration

Reference config at `configs/g2p.yaml`:

```yaml
binarizer:
  g2p:
    preprocessors:
      - id: filter-punctuation
      - id: strip-whitespace
      - id: lowercase
    tokenizers:
      - id: whitespace
      - id: cjk
    converters:
      # Chinese: hanzi
      - id: chinese-pinyin
        kwargs:
          dict_path: "dictionaries/ds-zh-pinyin-lite.txt"
      # Chinese: pinyin (direct dictionary fallback)
      - id: dictionary
        language: zh
        kwargs:
          dict_path: "dictionaries/ds-zh-pinyin-lite.txt"
      # Japanese: kana
      - id: japanese-kana
        kwargs:
          dict_path: "dictionaries/japanese_dict_full.txt"
          double_written_sokuon: false
      # Japanese: romaji (direct dictionary fallback)
      - id: dictionary
        language: ja
        kwargs:
          dict_path: "dictionaries/japanese_dict_full.txt"
      # English: dictionary + LSTM OOV
      - id: lstm
        language: en
        kwargs:
          dict_path: "dictionaries/ds_cmudict-07b.txt"
          model_path: "assets/LstmG2p-Eng"
```

### Config schema

```python
class G2PPipelineConfig(ConfigBaseModel):
    preprocessors: list[PreprocessorConfig]
    tokenizers: list[TokenizerConfig]
    converters: list[ConverterConfig]

class TokenizerConfig(ConfigBaseModel):
    id: str
    kwargs: dict

class PreprocessorConfig(ConfigBaseModel):
    id: str
    kwargs: dict

class ConverterConfig(ConfigBaseModel):
    id: str
    language: str | None = None   # comma-separated, overrides decorator default
    kwargs: dict
```

`language` on `ConverterConfig` is **required** when the converter class has no registered language (`cls.language is None`). It overrides the decorator default when set.

`@`-prefixed strings in kwargs resolve relative to `root_path`: `"@../dicts/eng.txt"` with `root_path="configs/g2p/"` → `configs/dicts/eng.txt`.

## Programmatic usage

```python
from g2p import G2PPipeline
from g2p.tokenizers.simple import WhitespaceTokenizer
from g2p.preprocessors.simple import LowercasePreprocessor
from g2p.converters.dictionary import DictionaryConverter
from g2p.converters.simple import PassthroughConverter

pipeline = G2PPipeline(
    preprocessors=[LowercasePreprocessor()],
    tokenizers=[WhitespaceTokenizer()],
    converters=[DictionaryConverter(path="dict.txt"), PassthroughConverter()],
)

result = pipeline.convert("Hello world")
# → [PronunciationGroup(paths=[["hh","ax","l","ow"],["hh","eh","l","ow"]], language=None),
#    PronunciationGroup(paths=[["w","er","l","d"]], language=None)]
```

Language filtering:

```python
result = pipeline.convert("你好", languages=["cmn"])
# MandarinConverter matched with "cmn"
# → [PronunciationGroup(paths=[["ni"]], language="cmn")]
```

Config-based:

```python
from lib.config.schema import G2PPipelineConfig
from g2p.api import build_pipeline_from_config

config = G2PPipelineConfig.model_validate({...})
pipeline = build_pipeline_from_config(config, root_path="configs/")
```

## Adding a custom converter

Drop a file into `g2p/converters/`:

```python
# g2p/converters/my_lang.py
from g2p.registry import converter
from g2p.converters.base import Converter, PronunciationGroup

@converter(id="my-lang", language="xyz")
class MyLangConverter(Converter):
    def claim(self, token: str) -> bool:
        return True

    def convert(self, tokens: list[str]) -> list[PronunciationGroup]:
        return [PronunciationGroup(paths=[[...]]) for t in tokens]
```

Or derive from a paradigm:

```python
from g2p.converters.paradigm import PronunciationScriptConverter

class MyConverter(PronunciationScriptConverter):
    def text_to_script(self, tokens): ...
    def script_to_phonemes(self, script): ...
```

Then reference in config:

```yaml
converters:
  - id: my-lang
    kwargs:
      some_option: false
```

No other files need editing.
