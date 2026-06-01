import pathlib
from typing import Any

import lightning.pytorch.callbacks
import textgrid


class SaveTextGridCallback(lightning.pytorch.callbacks.Callback):
    """Writes 2-tier TextGrid files from forced alignment results.

    Tiers:
    - words: intervals from decoded spans, labels from lexicon
    - phones: intervals from decoded spans, labels from G2P output
    """

    def __init__(
            self,
            output_dir: str | pathlib.Path,
            language: str | None = None,
    ):
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.language = language

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: list[dict[str, Any]],
            batch: dict[str, Any],
            *args, **kwargs,
    ) -> None:
        for result in outputs:
            identifier = result["identifier"]
            spans = result["spans"].tolist()  # [[onset, offset], ...]
            words = result["words"].tolist()  # [group_id, ...]
            phonemes = result["phonemes"]  # [str, ...]
            lexicon = result["lexicon"]

            N = len(spans)
            if N == 0:
                continue

            total_duration = result["duration"]
            tg = textgrid.TextGrid()

            # Words tier: group phoneme spans by consecutive G2PText index and
            # reverse-lookup the phoneme subsequence in the lexicon.
            words_tier = textgrid.IntervalTier("words", 0, total_duration)
            i = 0
            while i < N:
                w = words[i] - 1  # 0-based G2PText index
                j = i + 1
                while j < N and words[j] - 1 == w:
                    j += 1
                onset = spans[i][0]
                offset = spans[j - 1][1]
                word_text = lexicon[w].get(tuple(phonemes[i:j]), "") if w < len(lexicon) else ""
                words_tier.add(onset, offset, word_text)
                i = j
            tg.append(words_tier)

            # Phones tier
            phones_tier = textgrid.IntervalTier("phones", 0, total_duration)
            for n in range(N):
                onset, offset = spans[n]
                phones_tier.add(onset, offset, phonemes[n])
            tg.append(phones_tier)

            output_path = self.output_dir / f"{identifier}.TextGrid"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tg.write(str(output_path))
