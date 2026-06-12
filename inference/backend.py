import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lib.config.schema import InferenceConfig, ModelConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.path_traversal import compact_sequences, extract_tokens
from lib.vocabulary import MASK_TOKEN, SPACE_TOKEN
from modules.decoding import decode_alignment_flat
from modules.forced_alignment import ForcedAlignmentModel, ForcedAlignmentSSLModel
from modules.functional import cross_cosine_similarity


@dataclass
class SpectrogramContext:
    """Batched mel spectrogram with a validity mask."""
    features: Tensor  # [B, T, C]
    mask: Tensor  # [B, T] bool, True = valid frame


@dataclass
class ScoreResult:
    """Output of pronunciation scoring / disambiguation."""
    tokens: Tensor  # [B, N] best token sequence (compacted, 0-padded)
    groups: Tensor  # [B, N] best group sequence (compacted, 0-padded)
    alts: Tensor  # [B, S_max] chosen alternative index per segment
    scores: Tensor  # [B, S_max, W_max] per-segment mean log-likelihood-ratio (0-padded for invalid alternatives)


@dataclass
class AlignResult:
    """Output of forced alignment decoding."""
    spans: Tensor  # [B, N, 2] in the requested unit (seconds or frames)
    similarity: Tensor  # [B, T, N] cross cosine similarity


class InferenceBackend(ABC):
    """Protocol for forced alignment inference backends.

    Two-method design:
    - score(): score pronunciation alternatives and pick the best path
    - align(): produce forced alignment for a token sequence
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate."""

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Seconds per frame."""

    @abstractmethod
    def spectrogram(
            self, waveform: Tensor, duration: Tensor,
    ) -> SpectrogramContext:
        """Compute mel spectrogram and frame validity mask.

        Args:
            waveform: ``[B, L]`` float audio samples, 0-padded.
            duration: ``[B]`` float seconds of actual audio per item.

        Returns:
            SpectrogramContext with ``features [B,T,C]`` and ``mask [B,T]``.
        """

    @abstractmethod
    def score(
            self,
            spec: SpectrogramContext,
            *,
            paths: Tensor,
            groups: Tensor,
            segments: Tensor,
            widths: Tensor,
    ) -> ScoreResult:
        """Score pronunciation alternatives and pick the best path.

        For single-path items this is a no-op that returns the sole path.

        Args:
            spec: SpectrogramContext.
            paths: ``[B, N_grid, W_max]`` int64, alternative token IDs.
            groups: ``[B, N_grid, W_max]`` int64, alternative group IDs.
            segments: ``[B, N_grid]`` int64, segment index per grid position.
            widths: ``[B, S_max]`` int64, number of alternatives per segment.

        Returns:
            ScoreResult with the best token sequence and per-segment scores.
        """

    @abstractmethod
    def align(
            self,
            spec: SpectrogramContext,
            *,
            tokens: Tensor,
            groups: Tensor | None = None,
            unit: Literal["frame", "second"] = "second",
    ) -> AlignResult:
        """Produce forced alignment for a token sequence.

        Args:
            spec: SpectrogramContext.
            tokens: ``[B, N]`` int64 phoneme token IDs, 0-padded.
            groups: ``[B, N]`` int64 group IDs or None (each token = own group).
            unit: ``"second"`` returns spans in seconds (default);
                ``"frame"`` returns spans as frame indices.

        Returns:
            AlignResult with spans in the requested unit and similarity matrix.
        """


class ForcedAlignmentInferenceModel(nn.Module, InferenceBackend):
    """Supervised FA inference with MLM-based pronunciation scoring."""

    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
    ):
        super().__init__()
        feat = inference_config.features
        self._timestep = feat.hop_size / feat.audio_sample_rate
        self.spec_fn = StretchableMelSpectrogram(
            sample_rate=feat.audio_sample_rate,
            n_mels=feat.spectrogram.num_bins,
            n_fft=feat.fft_size,
            win_length=feat.win_size,
            hop_length=feat.hop_size,
            fmin=feat.spectrogram.fmin,
            fmax=feat.spectrogram.fmax,
        )
        self.model = ForcedAlignmentModel(model_config)

    @property
    def timestep(self) -> float:
        return self._timestep

    @property
    def sample_rate(self) -> int:
        return self.spec_fn.sample_rate

    def spectrogram(
            self, waveform: Tensor, duration: Tensor,
    ) -> SpectrogramContext:
        features = self.spec_fn(waveform).permute(0, 2, 1)  # [B, T, C]
        T = features.shape[1]
        L = duration.div(self.timestep).round().long()  # [B]
        idx = torch.arange(T, dtype=torch.long, device=duration.device)
        mask = idx.unsqueeze(0) < L.unsqueeze(1)  # [B, T]
        return SpectrogramContext(features=features, mask=mask)

    def score(
            self,
            spec: SpectrogramContext,
            *,
            paths: Tensor,
            groups: Tensor,
            segments: Tensor,
            widths: Tensor,
    ) -> ScoreResult:
        device = spec.features.device
        B = paths.shape[0]
        S_max = widths.shape[1]
        W_max = paths.shape[2]

        single_path = widths.max(dim=1).values == 1  # [B]
        multi_idx = torch.nonzero(~single_path, as_tuple=True)[0]  # [B_m]
        B_m = int(multi_idx.numel())

        best_alts = torch.zeros(B, S_max, dtype=torch.int64, device=device)
        mean_scores = torch.zeros(B, S_max, W_max, device=device)

        if B_m > 0:
            paths_m = paths[multi_idx]
            segments_m = segments[multi_idx]
            widths_m = widths[multi_idx]
            features_m = spec.features[multi_idx]
            mask_m = spec.mask[multi_idx]

            # ---- Phase A: Build masked token sequence ----
            N_total = int((segments_m != 0).sum(dim=-1).max().item())
            masked_tokens = paths_m[:, :N_total, 0].clone()

            widths_0 = torch.cat([
                torch.ones(B_m, 1, dtype=torch.int64, device=device), widths_m,
            ], dim=1)  # [B_m, 1 + S_max]
            div_mask = widths_0.gather(1, segments_m[:, :N_total]) > 1
            masked_tokens[div_mask] = MASK_TOKEN

            seq_lens = (segments_m != 0).sum(dim=-1)  # [B_m]
            pos_mask = (
                    torch.arange(N_total, device=device).unsqueeze(0)
                    >= seq_lens.unsqueeze(-1)
            )
            masked_tokens[pos_mask] = 0

            # ---- Phase B: MLM forward ----
            n_mask_mlm = masked_tokens != 0
            _, _, _, token_logits = self.model(
                features_m, masked_tokens, mask_m, n_mask_mlm,
            )

            # ---- Phase C: Score queries and aggregate per segment ----
            V = token_logits.shape[-1]
            log_probs = F.log_softmax(token_logits.float(), dim=-1)

            s_g = segments_m[:, :N_total].unsqueeze(-1) - 1  # [B_m, N_total, 1]
            w_at_pos = widths_0.gather(1, segments_m[:, :N_total]).unsqueeze(-1)
            w_g = torch.arange(W_max, device=device).view(1, 1, -1)

            valid_g = (s_g >= 0) & (w_g < w_at_pos) & (w_at_pos > 1)
            flat_g = valid_g.nonzero(as_tuple=False)  # [num_valid, 3]
            b_f, r_f, w_f = flat_g[:, 0], flat_g[:, 1], flat_g[:, 2]
            tok_f = paths_m[b_f, r_f, w_f]

            queries = torch.zeros(
                B_m, W_max, N_total, dtype=torch.int64, device=device,
            )
            queries[b_f, w_f, r_f] = SPACE_TOKEN
            real = tok_f != 0
            queries[b_f[real], w_f[real], r_f[real]] = tok_f[real]

            expanded = log_probs.unsqueeze(1).expand(-1, W_max, -1, -1)
            scores = expanded.gather(
                -1, queries.unsqueeze(-1),
            ).squeeze(-1) + math.log(V)  # [B_m, W_max, N_total]
            q_mask = queries != 0

            seg_oh = (
                    segments_m[:, :N_total].unsqueeze(1) ==
                    torch.arange(1, S_max + 1, device=device).view(1, -1, 1)
            )  # [B_m, S_max, N_total]
            score_sum = (scores.unsqueeze(1) * seg_oh.unsqueeze(2)).sum(dim=-1)
            score_cnt = (
                    q_mask.unsqueeze(1) * seg_oh.unsqueeze(2)
            ).sum(dim=-1).clamp(min=1)
            m_scores = score_sum / score_cnt  # [B_m, S_max, W_max]

            alt_ok = (
                    (torch.arange(W_max, device=device).view(1, 1, -1)
                     < widths_m.unsqueeze(-1)) &
                    (widths_m.unsqueeze(-1) > 1)
            )
            m_scores[~alt_ok] = float("-inf")
            best_alts[multi_idx] = m_scores.argmax(dim=-1)
            m_scores[~alt_ok] = 0.0
            mean_scores[multi_idx] = m_scores

        # ---- Phase D: Extract best tokens for all items ----
        tokens_raw, groups_raw = extract_tokens(
            paths, groups, segments=segments, choices=best_alts,
        )
        tokens_best, groups_best = compact_sequences(tokens_raw, groups_raw)

        return ScoreResult(
            tokens=tokens_best,
            groups=groups_best,
            alts=best_alts,
            scores=mean_scores,
        )

    def align(
            self,
            spec: SpectrogramContext,
            *,
            tokens: Tensor,
            groups: Tensor | None = None,
            unit: Literal["frame", "second"] = "second",
    ) -> AlignResult:
        n_mask = tokens != 0
        frame_features, _, token_features, _ = self.model(
            spec.features, tokens, spec.mask, n_mask,
        )
        similarity = cross_cosine_similarity(frame_features, token_features)
        frame_lengths = spec.mask.sum(dim=-1)
        token_lengths = n_mask.sum(dim=-1)
        spans = decode_alignment_flat(
            similarity, frame_lengths, token_lengths, groups=groups,
        )
        if unit == "second":
            spans = spans.float() * self.timestep
        return AlignResult(
            spans=spans,
            similarity=similarity,
        )


class ForcedAlignmentSSLInferenceModel(nn.Module, InferenceBackend):
    """SSL FA inference: attention-based similarity + decode. STUB."""

    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
            **kwargs,
    ):
        super().__init__()
        feat = inference_config.features
        self._timestep = feat.hop_size / feat.audio_sample_rate
        self.spec_fn = StretchableMelSpectrogram(
            sample_rate=feat.audio_sample_rate,
            n_mels=feat.spectrogram.num_bins,
            n_fft=feat.fft_size,
            win_length=feat.win_size,
            hop_length=feat.hop_size,
            fmin=feat.spectrogram.fmin,
            fmax=feat.spectrogram.fmax,
        )
        self.model = ForcedAlignmentSSLModel(model_config)

    @property
    def timestep(self) -> float:
        return self._timestep

    @property
    def sample_rate(self) -> int:
        return self.spec_fn.sample_rate

    def spectrogram(
            self, waveform: Tensor, duration: Tensor,
    ) -> SpectrogramContext:
        features = self.spec_fn(waveform).permute(0, 2, 1)
        T = features.shape[1]
        L = duration.div(self.timestep).round().long()
        idx = torch.arange(T, dtype=torch.long, device=duration.device)
        mask = idx.unsqueeze(0) < L.unsqueeze(1)
        return SpectrogramContext(features=features, mask=mask)

    def score(self, spec, paths, groups, segments, widths):
        raise NotImplementedError("SSL inference not yet implemented")

    def align(self, spec, tokens, groups=None, unit: Literal["frame", "second"] = "second"):
        raise NotImplementedError("SSL inference not yet implemented")
