import lightning.pytorch as pl
import torch
from torch import Tensor

from inference.backend import InferenceBackend
from lib import logging
from lib.path_sampling import compact_sequences, extract_tokens, sample_paths_perm


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

        # ---- Step 1: sample paths ----
        choices = sample_paths_perm(widths, r=2)  # [B, K, S_max]
        K = choices.shape[1]

        # ---- Step 2: deduplicate choices ----
        item_idx = (
            torch.arange(B, device=device)
            .unsqueeze(1).expand(B, K).reshape(-1, 1)
        )  # [B*K, 1]
        choices_flat = choices.reshape(B * K, S_max)  # [B*K, S_max]
        stacked = torch.cat([item_idx, choices_flat], dim=-1)  # [B*K, 1+S_max]
        unique, inverse = torch.unique(stacked, dim=0, return_inverse=True)
        sample_idx = unique[:, 0]  # [U]
        unique_choices = unique[:, 1:]  # [U, S_max]
        U = int(unique_choices.shape[0])

        if U == 0:
            return []

        # ---- Step 3: extract tokens for unique choices ----
        paths_per_unique = paths[sample_idx]  # [U, N_grid_max, W_max_max]
        words_per_unique = words[sample_idx]  # [U, N_grid_max, W_max_max]
        segments_per_unique = segments[sample_idx]  # [U, N_grid_max]
        tokens_raw = extract_tokens(
            paths_per_unique, segments_per_unique, unique_choices,
        )  # [U, N_grid_max]
        words_raw = extract_tokens(
            words_per_unique, segments_per_unique, unique_choices,
        )  # [U, N_grid_max]

        compacted_tokens, compacted_words, compacted_segments = compact_sequences(
            tokens_raw, words_raw, segments_per_unique,
        )  # [U, N_max']

        # ---- Step 4: infer + score ----
        ctx = self.backend.infer(
            waveform[sample_idx], duration[sample_idx], compacted_tokens,
        )
        scores = self.backend.score(ctx)  # [U, N_max']

        T = torch.zeros(B, dtype=torch.long, device=device)
        T[sample_idx] = ctx.num_frames()  # [B]
        Lq = T * timestep  # [B]

        # ---- Step 5: scatter-reduce per-segment mean scores ----
        # Two-level averaging (first within each unique choice, then across
        # choices) is equivalent to global sum/count because for a fixed
        # (item, segment, alternative) the token positions are deterministic:
        # the same alternative always selects the same sub-path from the path
        # grid, so the token count per segment is constant across occurrences.
        seg_sum = (
            scores.new_zeros(U, S_max + 1)
            .scatter_add_(1, compacted_segments, scores)
        )[:, 1:]  # [U, S_max]
        seg_cnt = (
            scores.new_zeros(U, S_max + 1)
            .scatter_add_(1, compacted_segments, scores.new_ones(compacted_segments.shape))
        )[:, 1:]  # [U, S_max]
        seg_mean = seg_sum / seg_cnt.clamp(min=1)

        # ---- Step 6: aggregate per (item, segment, alternative) ----
        # Flatten (item, segment, alternative) to 1-d bins then scatter-add
        # votes from each unique choice.  We scatter both score and count to
        # compute the per-bin mean.
        W_max_val = int(widths.max().item())
        valid = seg_cnt > 0  # [U, S_max] -- segments that actually have tokens

        item_2d = sample_idx.unsqueeze(1).expand(-1, S_max)  # [U, S_max]
        seg_2d = torch.arange(S_max, device=device).unsqueeze(0).expand(U, -1)
        alt_2d = unique_choices  # [U, S_max]

        flat_idx = (
                item_2d * (S_max * W_max_val)
                + seg_2d * W_max_val
                + alt_2d
        )  # [U, S_max]

        flat_idx_valid = flat_idx[valid]
        seg_mean_valid = seg_mean[valid]

        acc_score = torch.zeros(B * S_max * W_max_val, device=device)
        acc_score.scatter_add_(0, flat_idx_valid, seg_mean_valid)

        acc_cnt = torch.zeros(B * S_max * W_max_val, device=device)
        acc_cnt.scatter_add_(0, flat_idx_valid, torch.ones_like(seg_mean_valid))

        acc_score = acc_score.reshape(B, S_max, W_max_val)
        acc_cnt = acc_cnt.reshape(B, S_max, W_max_val)
        avg_score = acc_score / acc_cnt.clamp(min=1)  # [B, S_max, W_max]
        best_alts = avg_score.argmax(dim=-1)  # [B, S_max]

        # ---- Step 7: classify items and decode ----
        # Real segments are those that received votes from at least one unique
        # choice (acc_cnt > 0).
        seg_valid = acc_cnt.sum(dim=-1) > 0  # [B, S_max]

        # For each item i, which of the K sampled paths agree with best_alts
        # on every real segment?  Padding segments are forced True.
        best_expanded = best_alts.unsqueeze(1)  # [B, 1, S_max]
        match = (
                (choices == best_expanded) | ~seg_valid.unsqueeze(1)
        ).all(dim=-1)  # [B, K]
        has_match = match.any(dim=1)  # [B]

        single_path = widths.max(dim=1).values == 1  # [B]
        pass1_mask = single_path | has_match  # [B]

        best_k = torch.where(
            single_path,
            torch.zeros(B, dtype=torch.long, device=device),
            match.float().argmax(dim=1),
        )  # [B]

        flat_best = torch.arange(B, device=device) * K + best_k  # [B]
        u_best_all = inverse[flat_best]  # [B]

        # --- Pass 1: batched decode for items whose best path was already inflected ---
        pass1_idx = pass1_mask.nonzero(as_tuple=True)[0]  # [P]
        pass1_local = torch.empty(B, dtype=torch.long, device=device)
        pass1_local[pass1_idx] = torch.arange(pass1_idx.numel(), device=device)

        spans_p1: Tensor | None = None
        tokens_p1: Tensor | None = None
        words_p1: Tensor | None = None
        if pass1_idx.numel() > 0:
            u_pass1 = u_best_all[pass1_idx]  # [P]
            spans_p1 = self.backend.decode(
                ctx[u_pass1], groups=compacted_words[u_pass1],
            )  # [P, N_max_batch, 2]
            tokens_p1 = compacted_tokens[u_pass1]  # [P, N_max']
            words_p1 = compacted_words[u_pass1]  # [P, N_max']

        # --- Pass 2: re-infer for items whose best path was not sampled ---
        pass2_idx = (~pass1_mask).nonzero(as_tuple=True)[0]  # [Q]
        pass2_local = torch.empty(B, dtype=torch.long, device=device)
        pass2_local[pass2_idx] = torch.arange(pass2_idx.numel(), device=device)

        spans_p2: Tensor | None = None
        tokens_p2: Tensor | None = None
        words_p2: Tensor | None = None
        if pass2_idx.numel() > 0:
            tokens_raw_p2 = extract_tokens(
                paths[pass2_idx],
                segments[pass2_idx],
                best_alts[pass2_idx],
            )  # [Q, N_grid_max]
            words_raw_p2 = extract_tokens(
                words[pass2_idx],
                segments[pass2_idx],
                best_alts[pass2_idx],
            )  # [Q, N_grid_max]

            tokens_p2, words_p2 = compact_sequences(
                tokens_raw_p2, words_raw_p2,
            )  # [Q, N_max'']

            ctx2 = self.backend.infer(
                waveform[pass2_idx], duration[pass2_idx], tokens_p2,
            )
            spans_p2 = self.backend.decode(ctx2, groups=words_p2)

        # ---- Results ----
        results = []
        for i in range(B):
            # Find the pass that contains i
            if pass1_mask[i]:
                j, spans_px, tokens_px, words_px = (
                    pass1_local[i], spans_p1, tokens_p1, words_p1,
                )
            else:
                j, spans_px, tokens_px, words_px = (
                    pass2_local[i], spans_p2, tokens_p2, words_p2,
                )

            L_i = Lq[i].item()
            N_i = int((tokens_px[j] != 0).sum().item())
            spans_i = spans_px[j, :N_i]
            tokens_i = tokens_px[j, :N_i]
            words_i = words_px[j, :N_i]

            phs: list[str] = []
            pm = phonemes[i]
            for s, alt in enumerate(best_alts[i].tolist()):
                phs.extend(pm.get((s, alt), []))

            results.append({
                "identifier": batch["identifier"][i],
                "duration": L_i,
                "tokens": tokens_i,
                "words": words_i,
                "spans": spans_i,
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
