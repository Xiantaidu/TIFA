import json
import pathlib
from typing import Any

import lightning.pytorch.callbacks
import matplotlib.pyplot as plt
import textgrid
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import nn

from lib.plot import alignment_to_figure, cross_similarity_to_figure, topk_bar_figure
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
    """Owns metric instances, converts spans, computes + exports JSON.

    ``unit`` is ``"frame"`` or ``"ms"`` and determines the conversion
    factor applied to raw spans before feeding metrics.

    ``vocab`` provides ``.vocab_size`` and ``.decode(token_id)`` for
    per-token statistics.
    """

    def __init__(
            self,
            unit: str,
            vocab: "Vocabulary",
            ber_tols: list[int],
            token_topk: list[int],
            pair_topk: list[int],
            save_path: pathlib.Path,
            plot: bool = False,
    ):
        super().__init__()
        if unit == "frame":
            self._unit_factor = 1.0
        elif unit == "ms":
            self._unit_factor = 1000.0
        else:
            raise ValueError(f"Unknown unit: {unit}")
        self.unit = unit
        self.save_path = pathlib.Path(save_path)
        self.plot = plot
        self.ber_tols = ber_tols
        self.token_topk = token_topk
        self.pair_topk = pair_topk
        self._vocab = vocab
        self._results: dict | None = None

        metrics: dict[str, nn.Module] = {}
        max_k = max(token_topk) if token_topk else 0
        for tol in ber_tols:
            for mode in ("onset", "offset", "both"):
                metrics[f"ber/{mode}/{tol}"] = BoundaryErrorRate(
                    tolerance=tol, mode=mode,
                )
            if max_k:
                for mode in ("onset", "offset"):
                    metrics[f"ber/{mode}/{tol}/{max_k}"] = BoundaryErrorRate(
                        tolerance=tol, mode=mode,
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
        spans_pred_ms = outputs["spans"].float() * self._unit_factor
        spans_gt_ms = batch["spans"].float() * self._unit_factor
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

        @rank_zero_only
        def _save_plots():
            if self.plot:
                self._save_statistic_plots()

        _save_summary()
        _save_plots()

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
        for tol in self.ber_tols:
            for mode in ("onset", "offset", "both"):
                ber_variants.append({
                    "arguments": {"mode": mode, "tolerance": tol},
                    "value": float(self.metrics[f"ber/{mode}/{tol}"].compute().item()),
                })
            if self._vocab and self.token_topk:
                max_k = max(self.token_topk)
                for mode in ("onset", "offset"):
                    m = self.metrics[f"ber/{mode}/{tol}/{max_k}"]
                    ber_variants.append({
                        "arguments": {"mode": mode, "tolerance": tol, "k": max_k},
                        "value": float(m.compute().item()),
                    })
                    top = m.compute_top_k()
                    if top:
                        ber_statistics.append({
                            "arguments": {"mode": mode, "tolerance": tol},
                            "groups": [
                                {"key": self._vocab.decode(tid), "value": float(v.item())}
                                for tid, v in sorted(top.items(), key=lambda x: -x[1])
                            ],
                        })
        metrics_list.append({
            "name": "BER",
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
            "name": "B-MAE", "unit": self.unit,
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
        if self._vocab and self.token_topk:
            max_k = max(self.token_topk)
            ov_metric = self.metrics[f"overlap/{max_k}"]
            top = ov_metric.compute_top_k()
            if top:
                for metric_name in (f"overlap_precision@{max_k}", f"overlap_recall@{max_k}"):
                    if metric_name in top and top[metric_name]:
                        ov_statistics.append({
                            "arguments": {"k": max_k, "metric": metric_name.split("_", 1)[1].split("@")[0]},
                            "groups": [
                                {"key": self._vocab.decode(tid), "value": float(v.item())}
                                for tid, v in sorted(
                                    top[metric_name].items(), key=lambda x: -x[1],
                                )
                            ],
                        })
        metrics_list.append({
            "name": "Overlap",
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
        if self._vocab and self.pair_topk:
            max_k = max(self.pair_topk)
            cm = self.metrics[f"conj_mae/{max_k}"]
            top = cm.compute_top_k()
            if top:
                cm_statistics.append({
                    "arguments": {},
                    "groups": [
                        {
                            "key": f"{self._vocab.decode(ti, stringfy=True)},{self._vocab.decode(tj, stringfy=True)}",
                            "value": float(v.item()),
                        }
                        for (ti, tj), v in sorted(
                            top.items(), key=lambda x: -x[1],
                        )
                    ],
                })
        metrics_list.append({
            "name": "Conj-MAE", "unit": self.unit,
            "variants": cm_variants, "statistics": cm_statistics,
        })

        return {"metrics": metrics_list}

    def _save_statistic_plots(self) -> None:
        """Save top-k bar chart figures for metrics with largest k."""
        save_dir = self.save_path.parent / "statistics"
        save_dir.mkdir(parents=True, exist_ok=True)
        K = max(self.token_topk) if self.token_topk else 0
        KC = max(self.pair_topk) if self.pair_topk else 0

        for tol in self.ber_tols:
            for mode in ("onset", "offset"):
                if K:
                    self._plot_boundary_topk(
                        self.metrics[f"ber/{mode}/{tol}/{K}"],
                        key=f"ber/{mode}/{tol}/{K}", save_dir=save_dir,
                    )
        if K:
            for mode in ("onset", "offset"):
                self._plot_boundary_topk(
                    self.metrics[f"b_mae/{mode}/{K}"],
                    key=f"b_mae/{mode}/{K}", save_dir=save_dir,
                )
            self._plot_overlap_topk(
                self.metrics[f"overlap/{K}"],
                key=f"overlap/{K}", save_dir=save_dir,
            )
        if KC:
            self._plot_conjunction_topk(
                self.metrics[f"conj_mae/{KC}"],
                key=f"conj_mae/{KC}", save_dir=save_dir,
            )

    def _plot_boundary_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        labels = [
            self._vocab.decode(tid, stringfy=True) or str(tid)
            for tid in top
        ]
        values = [v.item() for v in top.values()]
        safe_key = key.replace("/", "_")
        fig = topk_bar_figure(labels, values, key)
        fig.savefig(save_dir / f"{safe_key}.jpg")
        plt.close(fig)

    def _plot_overlap_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        safe_key = key.replace("/", "_")
        for sub_name, sub_data in top.items():
            labels = [
                self._vocab.decode(tid, stringfy=True) or str(tid)
                for tid in sub_data
            ]
            values = [v.item() for v in sub_data.values()]
            safe_sub = f"{safe_key}_{sub_name.replace('/', '_')}"
            fig = topk_bar_figure(labels, values, f"{key} {sub_name}", reverse=False)
            fig.savefig(save_dir / f"{safe_sub}.jpg")
            plt.close(fig)

    def _plot_conjunction_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        labels = [
            f"{self._vocab.decode(i, stringfy=True) or str(i)} -> "
            f"{self._vocab.decode(j, stringfy=True) or str(j)}"
            for (i, j) in top
        ]
        values = [v.item() for v in top.values()]
        safe_key = key.replace("/", "_")
        fig = topk_bar_figure(labels, values, key)
        fig.savefig(save_dir / f"{safe_key}.jpg")
        plt.close(fig)


class VisualizeAlignmentCallback(lightning.pytorch.callbacks.Callback):
    """Saves per-sample similarity and alignment plots during evaluation."""

    def __init__(self, save_dir: pathlib.Path, vocab: Vocabulary, num_digits: int, item_paths: list):
        super().__init__()
        self.save_dir = pathlib.Path(save_dir)
        self.vocab = vocab
        self._num_digits = num_digits
        self._item_paths = item_paths

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict,
            batch: dict,
            *args, **kwargs,
    ) -> None:
        spectrogram = batch["spectrogram"]
        similarity = outputs["similarity"]
        regions = batch["regions"]
        tokens = batch["tokens"]
        spans_gt = batch["spans"]
        spans_pred = outputs["spans"]
        indices = batch["indices"]
        T_all = batch["T"]
        N_all = batch["N"]

        B = indices.shape[0]
        for i in range(B):
            T_i = int(T_all[i].item())
            N_i = int(N_all[i].item())
            if T_i == 0 or N_i == 0:
                continue

            data_idx = int(indices[i].item())
            token_ids = tokens[i, :N_i].tolist()
            token_labels = [
                self.vocab.decode(int(tid), stringfy=True) or str(tid)
                for tid in token_ids
            ]

            item_path = self._item_paths[data_idx]
            name = str(data_idx).zfill(self._num_digits)

            # Similarity plot
            sim = similarity[i, :T_i, :N_i].float().detach().cpu().numpy()
            fig_sim = cross_similarity_to_figure(
                sim.T,  # [N_i, T_i] as expected by plot function
                regions=regions[i, :T_i].detach().cpu().numpy(),
                title=item_path,
                token_labels=token_labels,
            )
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig_sim.savefig(self.save_dir / f"{name}_sim.jpg")
            plt.close(fig_sim)

            # Alignment plot
            spec = spectrogram[i, :T_i].detach().cpu().numpy()
            ps = spans_pred[i, :N_i].detach().cpu().numpy()
            gs = spans_gt[i, :N_i].detach().cpu().numpy()
            fig_align = alignment_to_figure(
                spec,
                token_labels=token_labels,
                pred_spans=ps,
                gt_spans=gs,
                title=item_path,
            )
            fig_align.savefig(self.save_dir / f"{name}_align.jpg")
            plt.close(fig_align)
