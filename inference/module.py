import lightning.pytorch as pl
import torch

from inference.backend import InferenceBackend
from lib import logging
from lib.path_traversal import compact_sequences, extract_tokens
from lib.vocabulary import MASK_TOKEN, SPACE_TOKEN


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Forced alignment inference with MLM-based pronunciation scoring.

    Works with any InferenceBackend.
    """

    def __init__(self, backend: InferenceBackend):
        super().__init__()
        self.backend = backend

    def predict_step(self, batch, batch_idx):
        # Emit warnings collected by the dataset workers
        for msg in batch.get("warning", []):
            logging.warning(msg, callback=self.trainer.progress_bar_callback.print)

        if "waveform" not in batch:
            return []

        device = batch["waveform"].device
        timestep = self.backend.timestep
        B = len(batch["identifier"])

        # Tensors
        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B] seconds
        paths = batch["paths"]  # [B, N_grid, W_max]
        groups = batch["groups"]  # [B, N_grid, W_max]
        word_idx = batch["word_idx"]  # [B, N_grid, W_max]
        segments = batch["segments"]  # [B, N_grid]
        widths = batch["widths"]  # [B, S_max]

        # Non-tensor items
        phonemes = batch["phonemes"]  # list[dict[(int,int), list[str]]]
        words = batch["words"]  # list[list[str]]

        S_max = widths.shape[1]

        # Split: multi-path items go through MLM scoring; single-path skip it
        single_path = widths.max(dim=1).values == 1  # [B]
        multi_idx = torch.nonzero(~single_path, as_tuple=True)[0]  # [B_m]
        B_m = int(multi_idx.numel())

        best_alts = torch.zeros(B, S_max, dtype=torch.int64, device=device)

        if B_m > 0:
            # Slice to multi-path items only
            paths_m = paths[multi_idx]
            segments_m = segments[multi_idx]
            widths_m = widths[multi_idx]
            waveform_m = waveform[multi_idx]
            duration_m = duration[multi_idx]

            # ---- Step 1: Build one masked sequence per item ----
            N_total = int((segments_m != 0).sum(dim=-1).max().item())
            masked_tokens = paths_m[:, :N_total, 0].clone()

            # Prepend column of 1s to widths: index 0 (padding) -> width 1
            widths_0 = torch.cat([
                torch.ones(B_m, 1, dtype=torch.int64, device=device), widths_m,
            ], dim=1)  # [B_m, 1 + S_max]
            div_mask = widths_0.gather(1, segments_m[:, :N_total]) > 1
            masked_tokens[div_mask] = MASK_TOKEN

            # Pad beyond each item's actual length
            seq_lens = (segments_m != 0).sum(dim=-1)  # [B_m]
            pos_mask = (
                torch.arange(N_total, device=device).unsqueeze(0)
                >= seq_lens.unsqueeze(-1)
            )
            masked_tokens[pos_mask] = 0

            # ---- Step 2: MLM inference (one call) ----
            ctx_mlm = self.backend.infer(
                waveform_m, duration_m, masked_tokens,
            )

            # ---- Step 3: Score all alternatives in W_max queries ----
            # Query w packs every segment's w-th alt at its positions.
            W_max_val = paths_m.shape[2]
            w_g = torch.arange(W_max_val, device=device).view(1, 1, -1)

            # segment per grid position (0-based, -1 for padding)
            s_g = segments_m[:, :N_total].unsqueeze(-1) - 1  # [B_m, N_total, 1]
            w_at_pos = widths_0.gather(1, segments_m[:, :N_total]).unsqueeze(-1)

            valid_g = (s_g >= 0) & (w_g < w_at_pos) & (w_at_pos > 1)

            flat_g = valid_g.nonzero(as_tuple=False)  # [num_valid, 3]
            b_f, r_f, w_f = flat_g[:, 0], flat_g[:, 1], flat_g[:, 2]
            tok_f = paths_m[b_f, r_f, w_f]

            queries = torch.zeros(
                B_m, W_max_val, N_total, dtype=torch.int64, device=device,
            )
            queries[b_f, w_f, r_f] = SPACE_TOKEN
            real = tok_f != 0
            queries[b_f[real], w_f[real], r_f[real]] = tok_f[real]

            # One batched score call
            scores = self.backend.score(ctx_mlm, queries)  # [B_m, W_max_val, N_total]
            mask = queries != 0  # [B_m, W_max_val, N_total]

            # Per-segment: mask by segment via one-hot
            seg_oh = (
                segments_m[:, :N_total].unsqueeze(1) ==
                torch.arange(1, S_max + 1, device=device).view(1, -1, 1)
            )  # [B_m, S_max, N_total]
            score_sum = (scores.unsqueeze(1) * seg_oh.unsqueeze(2)).sum(dim=-1)
            score_cnt = (mask.unsqueeze(1) * seg_oh.unsqueeze(2)).sum(dim=-1).clamp(min=1)
            mean_scores = score_sum / score_cnt  # [B_m, S_max, W_max_val]

            # Mask invalid alts, argmax per segment
            alt_ok = (
                (torch.arange(W_max_val, device=device).view(1, 1, -1)
                 < widths_m.unsqueeze(-1)) &
                (widths_m.unsqueeze(-1) > 1)
            )
            mean_scores[~alt_ok] = float("-inf")
            best_alts[multi_idx] = mean_scores.argmax(dim=-1)  # [B_m, S_max]

        # ---- Phase D: Decode with best tokens ----
        tokens_raw, groups_raw, word_idx_raw = extract_tokens(
            paths, groups, word_idx, segments=segments, choices=best_alts,
        )  # [B, N_grid] each
        tokens_best, groups_best, word_idx_best = compact_sequences(
            tokens_raw, groups_raw, word_idx_raw,
        )  # [B, N_max']

        ctx_dec = self.backend.infer(waveform, duration, tokens_best)
        sim = self.backend.similarity(ctx_dec)  # [B, T_max, N_max']
        spans = self.backend.decode(ctx_dec, groups=groups_best)  # [B, N_max', 2]
        nf = ctx_dec.num_frames()  # [B]
        Lq = nf * timestep  # [B]

        # ---- Results ----
        results = []
        for i in range(B):
            N_i = int((tokens_best[i] != 0).sum().item())
            T_i = int(nf[i].item())
            spans_i = spans[i, :N_i]
            tokens_i = tokens_best[i, :N_i]
            groups_i = groups_best[i, :N_i]
            sim_i = sim[i, :T_i, :N_i]

            phs: list[str] = []
            pm = phonemes[i]
            for seg, alt in enumerate(best_alts[i].tolist()):
                phs.extend(pm.get((seg, alt), []))

            results.append({
                "identifier": batch["identifier"][i],
                "duration": Lq[i].item(),
                "tokens": tokens_i,
                "groups": groups_i,
                "word_idx": word_idx_best[i, :N_i],
                "spans": spans_i,
                "phonemes": phs,
                "words": words[i],
                "similarity": sim_i,
            })

        return results

    def test_step(self, batch, batch_idx) -> dict:
        """Online evaluation on PhonemeTimingDataset batches.

        Returns predicted spans in frames so they share the same unit
        as ground truth (batch["spans"]).  The callback then applies a
        single unit_size_ms to convert both to ms.
        """
        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B]
        tokens = batch["tokens"]      # [B, N]

        ctx = self.backend.infer(waveform, duration, tokens)
        similarity = self.backend.similarity(ctx)  # [B, T, N]
        spans_pred = self.backend.decode(ctx)  # [B, N, 2] seconds

        # Convert to frames so unit matches batch["spans"]
        spans_pred = spans_pred / self.backend.timestep

        result = {
            "spans": spans_pred,
            "similarity": similarity
        }
        return result


class OfflineEvaluationModule(pl.LightningModule):
    """Passes paired TextGrid spans to callbacks.

    Batch (from PairedDataset) contains both spans_pred and spans_gt.
    Returns batch["spans_pred"] so the callback sees outputs["spans"].
    """

    def test_step(self, batch, batch_idx) -> dict:
        return {"spans": batch["spans_pred"]}
