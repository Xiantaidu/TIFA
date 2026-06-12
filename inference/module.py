import lightning.pytorch as pl
import torch

from inference.backend import InferenceBackend, SpectrogramContext
from lib import logging


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Forced alignment inference. Works with any InferenceBackend."""

    def __init__(self, backend: InferenceBackend):
        super().__init__()
        self.backend = backend

    def predict_step(self, batch, batch_idx):
        for msg in batch.get("warning", []):
            logging.warning(msg, callback=self.trainer.progress_bar_callback.print)

        if "waveform" not in batch:
            return []

        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B] seconds
        paths = batch["paths"]  # [B, N_grid, W_max]
        groups = batch["groups"]  # [B, N_grid, W_max]
        segments = batch["segments"]  # [B, N_grid]
        widths = batch["widths"]  # [B, S_max]
        phonemes = batch["phonemes"]  # list[dict[(int,int), list[str]]]
        lexicon = batch["lexicon"]  # list[list[dict[str, list[list[str]]]]]

        # ---- Spectrogram ----
        spec = self.backend.spectrogram(waveform, duration)

        # ---- Score ----
        scored = self.backend.score(spec, paths=paths, groups=groups, segments=segments, widths=widths)

        # ---- Align ----
        result = self.backend.align(spec, tokens=scored.tokens, groups=scored.groups)

        # ---- Assemble per-item results ----
        results = []
        for i, identifier in enumerate(batch["identifier"]):
            N_i = int((scored.tokens[i] != 0).sum().item())
            T_i = int(spec.mask[i].sum().item())
            spans_i = result.spans[i, :N_i]
            tokens_i = scored.tokens[i, :N_i]
            groups_i = scored.groups[i, :N_i]
            sim_i = result.similarity[i, :T_i, :N_i]

            phs: list[str] = []
            pm = phonemes[i]
            for seg, alt in enumerate(scored.alts[i].tolist()):
                phs.extend(pm.get((seg, alt), []))

            results.append({
                "identifier": identifier,
                "duration": T_i * self.backend.timestep,
                "spans": spans_i,
                "tokens": tokens_i,
                "groups": groups_i,
                "similarity": sim_i,
                "phonemes": phs,
                "lexicon": lexicon[i],
            })

        return results

    def test_step(self, batch, batch_idx) -> dict:
        """Online evaluation on PhonemeTimingDataset batches.

        Returns predicted spans in frames so they share the same unit
        as ground truth (batch["spans"]).  The callback then applies a
        single unit_size_ms to convert both to ms.
        """
        spectrogram = batch["spectrogram"]  # [B, T, C]
        tokens = batch["tokens"]  # [B, N]
        T = spectrogram.shape[1]
        device = spectrogram.device

        mask = torch.arange(T, device=device).unsqueeze(0) < batch["T"].unsqueeze(1)
        spec = SpectrogramContext(features=spectrogram, mask=mask)
        result = self.backend.align(spec, tokens=tokens)

        spans_pred = result.spans / self.backend.timestep  # seconds to frames
        return {"spans": spans_pred, "similarity": result.similarity}


class OfflineEvaluationModule(pl.LightningModule):
    """Passes paired TextGrid spans to callbacks.

    Batch (from PairedDataset) contains both spans_pred and spans_gt.
    Returns batch["spans_pred"] so the callback sees outputs["spans"].
    """

    def test_step(self, batch, batch_idx) -> dict:
        return {"spans": batch["spans_pred"]}
