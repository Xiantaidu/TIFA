import json
import pathlib
from typing import Any

import lightning.pytorch.callbacks
import textgrid
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import nn

from modules.metrics import (
    BoundaryErrorRate,
    BoundaryMAE,
    OverlapRatioCollection,
    PairConjunctionMAE,
)


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


class EvaluationMetricsCallback(lightning.pytorch.callbacks.Callback):
    """Owns metric instances, converts spans to ms, computes + exports JSON.

    Unit conversion: spans_pred and spans_gt are both multiplied by
    ``unit_size_ms`` before feeding to metrics.  For online evaluation
    ``unit_size_ms = hop_size / sample_rate * 1000`` (frames -> ms);
    for offline ``unit_size_ms = 1000`` (seconds -> ms).
    """

    def __init__(
            self,
            unit_size_ms: float,
            vocab_size: int,
            ber_tolerance_ms: float,
            k_values: list[int],
            conjunction_k_values: list[int],
            save_path: pathlib.Path,
    ):
        super().__init__()
        self.unit_size_ms = unit_size_ms
        self.save_path = pathlib.Path(save_path)
        self._results: dict[str, float] | None = None

        metrics: dict[str, nn.Module] = {}
        for mode in ("onset", "offset", "both"):
            metrics[f"boundary_error_rate_{mode}"] = BoundaryErrorRate(
                tolerance=ber_tolerance_ms, mode=mode,
            )
        for k in k_values:
            for mode in ("onset", "offset"):
                metrics[f"boundary_mae_{mode}@{k}"] = BoundaryMAE(
                    mode=mode, vocab_size=vocab_size, k=k,
                )
            metrics[f"overlap_precision@{k}"] = OverlapRatioCollection(
                template=f"overlap_{{}}@{k}", vocab_size=vocab_size, k=k,
            )
        for k in conjunction_k_values:
            metrics[f"pair_conjunction_mae@{k}"] = PairConjunctionMAE(
                vocab_size=vocab_size, k=k,
            )
        self.metrics = nn.ModuleDict(metrics)

    def on_test_start(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self.metrics.to(trainer.strategy.root_device)

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict,
            batch: dict,
            *args, **kwargs,
    ) -> None:
        spans_pred_ms = outputs["spans"].float() * self.unit_size_ms
        spans_gt_ms = batch["spans"].float() * self.unit_size_ms
        tokens = batch["tokens"]
        for metric in self.metrics.values():
            metric.update(spans_pred_ms, spans_gt_ms, tokens)

    def on_test_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self._results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                for k, v in value.items():
                    self._results[k] = float(v.item())
            else:
                self._results[name] = float(value.item())

        @rank_zero_only
        def _save_summary():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w", encoding="utf8") as f:
                json.dump(self._results, f, indent=2)

        _save_summary()
