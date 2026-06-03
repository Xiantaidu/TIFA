import lightning.pytorch as pl
import torch

from inference.backend import InferenceBackend
from lib import logging
from lib.path_traversal import (
    accumulate_best_alts,
    build_length_grid,
    compact_sequences,
    extract_tokens,
    gather_query_tokens,
    layout_segments,
    sample_paths_perm,
    scatter_real_tokens,
    segment_grid_starts,
)
from lib.vocabulary import MASK_TOKEN


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Two-pass forced alignment inference with re-inference skip.

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

        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B] seconds

        paths = batch["paths"]  # [B, N_grid_max, W_max_max]
        words = batch["words"]  # [B, N_grid_max]
        segments = batch["segments"]  # [B, N_grid_max]
        widths = batch["widths"]  # [B, S_max]
        phonemes = batch["phonemes"]  # list[dict[(int,int), list[str]]]
        lexicon = batch["lexicon"]  # pass-through to callbacks

        S_max = widths.shape[1]
        W_max_val = int(widths.max().item())

        # Split: multi-path items go through MLM scoring; single-path skip it
        single_path = widths.max(dim=1).values == 1  # [B]
        multi_idx = torch.nonzero(~single_path, as_tuple=True)[0]  # [B_m]
        B_m = int(multi_idx.numel())

        best_alts = torch.zeros(B, S_max, dtype=torch.int64, device=device)

        if B_m > 0:
            # Slice to multi-path items only
            paths_m = paths[multi_idx]  # [B_m, N_grid_max, W_max]
            segments_m = segments[multi_idx]  # [B_m, N_grid_max]
            widths_m = widths[multi_idx]  # [B_m, S_max]
            waveform_m = waveform[multi_idx]
            duration_m = duration[multi_idx]

            # ---- Phase A: Build length grid and sample ----
            length_widths, length_values, alt_lengths = build_length_grid(
                paths_m, segments_m, widths_m,
            )

            choices = sample_paths_perm(length_widths, r=2)  # [B_m, K, S_max]
            K = choices.shape[1]

            item_idx = (
                torch.arange(B_m, device=device)
                .unsqueeze(1).expand(B_m, K).reshape(-1, 1)
            )
            choices_flat = choices.reshape(B_m * K, S_max)
            stacked = torch.cat([item_idx, choices_flat], dim=-1)
            unique, _inverse = torch.unique(stacked, dim=0, return_inverse=True)
            sample_idx = unique[:, 0]  # [U] — indices into multi-path subset
            unique_lchoices = unique[:, 1:]  # [U, S_max]
            U = int(unique_lchoices.shape[0])

            # ---- Phase B: Build masked token sequences ----
            non_div_m = (widths_m <= 1)  # [B_m, S_max]
            non_div_u = non_div_m[sample_idx]  # [U, S_max]
            base_lens = alt_lengths[:, :, 0]  # [B_m, S_max]
            seg_lens, seg_starts, N_max_val = layout_segments(
                non_div=non_div_u,
                base_lens=base_lens,
                length_values=length_values,
                unique_lchoices=unique_lchoices,
                sample_idx=sample_idx,
            )

            seg_grid_starts = segment_grid_starts(segments_m, S_max)  # [B_m, S_max]

            masked_tokens = torch.full(
                (U, N_max_val), MASK_TOKEN, dtype=torch.int64, device=device,
            )
            pos_range = torch.arange(N_max_val, device=device).unsqueeze(0)
            masked_tokens[pos_range >= seg_lens.sum(dim=-1).unsqueeze(-1)] = 0

            scatter_real_tokens(
                target=masked_tokens,
                paths=paths_m,
                seg_starts=seg_starts,
                seg_lens=seg_lens,
                non_div_mask=non_div_u,
                item_idx=sample_idx,
                grid_starts=seg_grid_starts,
            )

            # ---- Phase C: MLM inference + scoring ----
            ctx_mlm = self.backend.infer(
                waveform_m[sample_idx], duration_m[sample_idx], masked_tokens,
            )

            alt_lens_per_u = alt_lengths[sample_idx]  # [U, S_max, W_max]
            seg_lens_exp = seg_lens.unsqueeze(-1)
            match_mask = (
                (alt_lens_per_u == seg_lens_exp)
                & ~non_div_u.unsqueeze(-1)
            )

            match_flat = match_mask.view(U, S_max * W_max_val)
            active_u, active_sw = match_flat.nonzero(as_tuple=True)
            Q = int(active_u.numel())
            active_s = active_sw // W_max_val
            active_w = active_sw % W_max_val
            active_sub_item = sample_idx[active_u]  # [Q] — index into multi-path subset

            q_u = active_u
            q_seg = active_s
            q_alt = active_w
            q_start = seg_starts[active_u, active_s]
            q_len = seg_lens[active_u, active_s]

            if Q > 0:
                q_local_flat = match_flat.int().cumsum(dim=-1) - 1
                q_local = q_local_flat[active_u, active_sw]

                queries_all = gather_query_tokens(
                    U=U, Q=Q, N=N_max_val,
                    paths=paths_m,
                    active_u=active_u,
                    active_s=active_s,
                    active_w=active_w,
                    active_item=active_sub_item,
                    q_start=q_start,
                    q_len=q_len,
                    q_local=q_local,
                    grid_starts=seg_grid_starts,
                )

                scores_all = self.backend.score(ctx_mlm, queries_all)
                q_mask = queries_all != 0
                q_scores = (
                    (scores_all * q_mask).sum(dim=-1)
                    / q_mask.sum(dim=-1).clamp(min=1)
                )

                best_alts_m = accumulate_best_alts(
                    q_scores=q_scores,
                    q_u=q_u,
                    q_local=q_local,
                    q_item=active_sub_item,
                    q_seg=q_seg,
                    q_alt=q_alt,
                    B=B_m,
                    S_max=S_max,
                    W_max=W_max_val,
                )
                best_alts[multi_idx] = best_alts_m

        # ---- Phase D: Decode with best tokens ----
        tokens_raw, words_raw = extract_tokens(
            paths, words, segments=segments, choices=best_alts,
        )  # [B, N_grid_max] each
        tokens_best, words_best = compact_sequences(
            tokens_raw, words_raw,
        )  # [B, N_max']

        ctx_dec = self.backend.infer(waveform, duration, tokens_best)
        sim = self.backend.similarity(ctx_dec)  # [B, T_max, N_max']
        spans = self.backend.decode(ctx_dec, groups=words_best)  # [B, N_max', 2]
        nf = ctx_dec.num_frames()  # [B]
        Lq = nf * timestep  # [B]

        # ---- Results ----
        results = []
        for i in range(B):
            N_i = int((tokens_best[i] != 0).sum().item())
            T_i = int(nf[i].item())
            spans_i = spans[i, :N_i]
            tokens_i = tokens_best[i, :N_i]
            words_i = words_best[i, :N_i]
            sim_i = sim[i, :T_i, :N_i]

            phs: list[str] = []
            pm = phonemes[i]
            for s, alt in enumerate(best_alts[i].tolist()):
                phs.extend(pm.get((s, alt), []))

            results.append({
                "identifier": batch["identifier"][i],
                "duration": Lq[i].item(),
                "tokens": tokens_i,
                "words": words_i,
                "spans": spans_i,
                "phonemes": phs,
                "lexicon": lexicon[i],
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
