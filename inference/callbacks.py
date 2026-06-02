import json
import pathlib
from typing import Any

import lightning.pytorch.callbacks
import textgrid
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import nn

from lib.vocabulary import Vocabulary
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

    ``vocab`` provides ``.vocab_size`` and ``.decode(token_id)`` for
    per-token statistics.
    """

    def __init__(
            self,
            unit_size_ms: float,
            vocab: "Vocabulary",
            ber_tols_ms: list[int],
            token_topk: list[int],
            pair_topk: list[int],
            save_path: pathlib.Path,
    ):
        super().__init__()
        self.unit_size_ms = unit_size_ms
        self.save_path = pathlib.Path(save_path)
        self.ber_tols_ms = ber_tols_ms
        self.token_topk = token_topk
        self.pair_topk = pair_topk
        self._vocab = vocab
        self._results: dict | None = None

        metrics: dict[str, nn.Module] = {}
        max_k = max(token_topk) if token_topk else 0
        for tol_ms in ber_tols_ms:
            for mode in ("onset", "offset", "both"):
                metrics[f"ber/{mode}/{tol_ms}"] = BoundaryErrorRate(
                    tolerance=tol_ms, mode=mode,
                )
            if max_k:
                for mode in ("onset", "offset"):
                    metrics[f"ber/{mode}/{tol_ms}/{max_k}"] = BoundaryErrorRate(
                        tolerance=tol_ms, mode=mode,
                        vocab_size=self._vocab.vocab_size, k=max_k,
                    )
        for k in token_topk:
            for mode in ("onset", "offset"):
                metrics[f"b_mae/{mode}/{k}"] = BoundaryMAE(
                    mode=mode, vocab_size=self._vocab.vocab_size, k=k,
                )
            metrics[f"overlap/{k}"] = OverlapRatioCollection(
                template=f"overlap_{{}}@{k}", vocab_size=self._vocab.vocab_size, k=k,
            )
        for k in pair_topk:
            metrics[f"conj_mae/{k}"] = PairConjunctionMAE(
                vocab_size=self._vocab.vocab_size, k=k,
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
        self._results = self._build_summary()

        @rank_zero_only
        def _save_summary():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w", encoding="utf8") as f:
                json.dump(self._results, f, indent=2)

        _save_summary()

    @property
    def results(self) -> dict | None:
        return self._results

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        metrics_list: list[dict] = []

        # -- BER --
        ber_variants: list[dict] = []
        ber_statistics: list[dict] = []
        for tol_ms in self.ber_tols_ms:
            for mode in ("onset", "offset", "both"):
                ber_variants.append({
                    "arguments": {"mode": mode, "tolerance": tol_ms},
                    "value": float(self.metrics[f"ber/{mode}/{tol_ms}"].compute().item()),
                })
            if self._vocab and self.token_topk:
                max_k = max(self.token_topk)
                for mode in ("onset", "offset"):
                    m = self.metrics[f"ber/{mode}/{tol_ms}/{max_k}"]
                    ber_variants.append({
                        "arguments": {"mode": mode, "tolerance": tol_ms, "k": max_k},
                        "value": float(m.compute().item()),
                    })
                    top = m.compute_top_k()
                    if top:
                        ber_statistics.append({
                            "arguments": {"mode": mode, "tolerance": tol_ms},
                            "groups": [
                                {"key": self._vocab.decode(tid), "value": float(v.item())}
                                for tid, v in sorted(top.items(), key=lambda x: -x[1])
                            ],
                        })
        metrics_list.append({
            "name": "BER", "unit": "ratio",
            "variants": ber_variants, "statistics": ber_statistics,
        })

        # -- B-MAE --
        b_mae_variants: list[dict] = []
        b_mae_statistics: list[dict] = []
        for mode in ("onset", "offset"):
            for k in self.token_topk:
                b_mae_variants.append({
                    "arguments": {"mode": mode, "k": k},
                    "value": float(self.metrics[f"b_mae/{mode}/{k}"].compute().item()),
                })
            if self._vocab:
                max_k = max(self.token_topk)
                m = self.metrics[f"b_mae/{mode}/{max_k}"]
                top = m.compute_top_k()
                if top:
                    b_mae_statistics.append({
                        "arguments": {"mode": mode},
                        "groups": [
                            {"key": self._vocab.decode(tid), "value": float(v.item())}
                            for tid, v in sorted(top.items(), key=lambda x: -x[1])
                        ],
                    })
        metrics_list.append({
            "name": "B-MAE", "unit": "ms",
            "variants": b_mae_variants, "statistics": b_mae_statistics,
        })

        # -- Overlap --
        ov_variants: list[dict] = []
        ov_statistics: list[dict] = []
        for k in self.token_topk:
            ov_metric = self.metrics[f"overlap/{k}"]
            vals = ov_metric.compute()
            ov_variants.append({
                "arguments": {"k": k},
                "value": {
                    "precision": float(vals[f"overlap_precision@{k}"].item()),
                    "recall": float(vals[f"overlap_recall@{k}"].item()),
                },
            })
            if self._vocab:
                top = ov_metric.compute_top_k()
                if top:
                    for metric_name in (f"overlap_precision@{k}", f"overlap_recall@{k}"):
                        if metric_name in top and top[metric_name]:
                            ov_statistics.append({
                                "arguments": {"k": k, "metric": metric_name.split("_", 1)[1].split("@")[0]},
                                "groups": [
                                    {"key": self._vocab.decode(tid), "value": float(v.item())}
                                    for tid, v in sorted(
                                        top[metric_name].items(), key=lambda x: -x[1],
                                    )
                                ],
                            })
        metrics_list.append({
            "name": "Overlap", "unit": "ratio",
            "variants": ov_variants, "statistics": ov_statistics,
        })

        # -- Conj-MAE --
        cm_variants: list[dict] = []
        cm_statistics: list[dict] = []
        for k in self.pair_topk:
            cm = self.metrics[f"conj_mae/{k}"]
            cm_variants.append({
                "arguments": {"k": k},
                "value": float(cm.compute().item()),
            })
            if self._vocab:
                top = cm.compute_top_k()
                if top:
                    cm_statistics.append({
                        "arguments": {},
                        "groups": [
                            {
                                "key": f"{self._vocab.decode(ti)},{self._vocab.decode(tj)}",
                                "value": float(v.item()),
                            }
                            for (ti, tj), v in sorted(
                                top.items(), key=lambda x: -x[1],
                            )
                        ],
                    })
        metrics_list.append({
            "name": "Conj-MAE", "unit": "ms",
            "variants": cm_variants, "statistics": cm_statistics,
        })

        return {"metrics": metrics_list}
