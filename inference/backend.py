import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lib.config.schema import InferenceConfig, ModelConfig
from lib.feature.mel import StretchableMelSpectrogram
from modules.decoding import decode_alignment_flat
from modules.forced_alignment import ForcedAlignmentModel, ForcedAlignmentSSLModel
from modules.functional import cross_cosine_similarity


class InferenceContext(ABC):
    """Opaque holder for model outputs. Backend-specific subclass.

    Must support indexing so the LightningModule can select a specific
    candidate path's context without re-running the model.
    """

    @abstractmethod
    def __getitem__(self, idx) -> "InferenceContext":
        ...

    @abstractmethod
    def num_frames(self) -> Tensor:
        """Number of valid spectrogram frames per item. ``[B]`` int64."""
        ...


class InferenceBackend(ABC):
    """Unified inference protocol for forced alignment models.

    Three-method design:
    - infer(): expensive model forward, returns cached context
    - score(): cheap, extracts per-token scores from context
    - decode(): cheap, decodes spans from context

    Separating these allows skipping re-inference when the best path
    was already evaluated in the first pass.
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
    def infer(
            self,
            waveform: Tensor,
            duration: Tensor,
            tokens: Tensor,
    ) -> InferenceContext:
        """Run model forward.

        Args:
            waveform: ``[B, L]`` float audio samples, 0-padded.
            duration: ``[B]`` float seconds of actual audio per item.
            tokens: ``[B, N]`` int64 phoneme token IDs, 0-padded.
        """

    @abstractmethod
    def similarity(self, ctx: InferenceContext) -> Tensor:
        """Cross similarity matrix ``[B, T, N]`` between frame and token features.

        Higher values indicate stronger alignment.
        """

    @abstractmethod
    def score(self, ctx: InferenceContext, queries: Tensor) -> Tensor:
        """Log-likelihood-ratio of each query token at each position.

        Computes ``log(p(token) * V)`` so that the uniform distribution
        gives a zero baseline.  An empty query row (all zeros) scores 0.
        Higher = better.

        Args:
            ctx: Inference context.
            queries: ``[B, Q, N]`` int64, candidate token IDs (0 = ignore).

        Returns:
            ``[B, Q, N]`` float.
        """

    @abstractmethod
    def decode(self, ctx: InferenceContext, groups: Tensor | None = None) -> Tensor:
        """Decode spans [B, N, 2] in seconds from context."""


@dataclass
class ForcedAlignmentContext(InferenceContext):
    frame_features: Tensor  # [B, T, C]
    token_features: Tensor  # [B, N, C]
    frame_logits: Tensor  # [B, T, V]
    token_logits: Tensor  # [B, N, V]
    tokens: Tensor  # [B, N] int64
    t_mask: Tensor  # [B, T] bool
    n_mask: Tensor  # [B, N] bool
    similarity: Tensor | None = None  # [B, T, N] cached by backend.similarity()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return ForcedAlignmentContext(
            frame_features=self.frame_features[idx],
            token_features=self.token_features[idx],
            frame_logits=self.frame_logits[idx],
            token_logits=self.token_logits[idx],
            tokens=self.tokens[idx],
            t_mask=self.t_mask[idx],
            n_mask=self.n_mask[idx],
            similarity=self.similarity[idx] if self.similarity is not None else None,
        )

    def num_frames(self) -> Tensor:
        return self.t_mask.sum(dim=-1)


class ForcedAlignmentInferenceModel(nn.Module, InferenceBackend):
    """Supervised FA inference.

    Scoring: top-k cosine-similarity frames per token, summed frame-CE
    softmax probability for the token's predicted phoneme class.
    """

    def __init__(
            self,
            model_config: ModelConfig,
            inference_config: InferenceConfig,
            topk: int = 10,
    ):
        super().__init__()
        self.topk = topk
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

    def infer(
            self,
            waveform: Tensor,
            duration: Tensor,
            tokens: Tensor,
    ) -> ForcedAlignmentContext:
        spectrogram = self.spec_fn(waveform).permute(0, 2, 1)  # [B, T, C]
        T = spectrogram.shape[1]
        L = duration.div(self.timestep).round().long()  # [B]
        idx = torch.arange(T, dtype=torch.long, device=duration.device)
        t_mask = idx.unsqueeze(0) < L.unsqueeze(1)  # [B, T]
        n_mask = tokens != 0  # [B, N]

        frame_features, frame_logits, token_features, token_logits = self.model(
            spectrogram, tokens, t_mask, n_mask,
        )
        return ForcedAlignmentContext(
            frame_features=frame_features,
            token_features=token_features,
            frame_logits=frame_logits,
            token_logits=token_logits,
            tokens=tokens,
            t_mask=t_mask,
            n_mask=n_mask,
        )

    def similarity(self, ctx: ForcedAlignmentContext) -> Tensor:
        if ctx.similarity is None:
            ctx.similarity = cross_cosine_similarity(
                ctx.frame_features, ctx.token_features,
            )
        return ctx.similarity

    def score(self, ctx: ForcedAlignmentContext, queries: Tensor) -> Tensor:
        """Log-likelihood-ratio of each query token at each position.

        Computes ``log(p(token) * V)`` where *V* is the vocabulary size,
        so that comparing to the uniform distribution gives a zero baseline:
        an empty query row scores 0 (neutral), a token scores >0 if the
        model predicts it better than random.

        Args:
            ctx: Inference context.
            queries: ``[B, Q, N]`` int64, candidate token IDs (0 = ignore).

        Returns:
            ``[B, Q, N]`` float, log-likelihood-ratio per position.
        """
        V = ctx.token_logits.shape[-1]
        log_probs = F.log_softmax(ctx.token_logits.float(), dim=-1)  # [B, N, V]
        Q = queries.shape[1]
        expanded = log_probs.unsqueeze(1).expand(-1, Q, -1, -1)  # [B, Q, N, V]
        return expanded.gather(-1, queries.unsqueeze(-1)).squeeze(-1) + math.log(V)

    def decode(self, ctx: ForcedAlignmentContext, groups: Tensor | None = None) -> Tensor:
        sim = self.similarity(ctx)
        frame_lengths = ctx.t_mask.sum(dim=-1)
        token_lengths = ctx.n_mask.sum(dim=-1)
        spans = decode_alignment_flat(sim, frame_lengths, token_lengths, groups=groups)
        return spans.float() * self.timestep


class ForcedAlignmentSSLInferenceModel(nn.Module, InferenceBackend):
    """SSL FA inference: attention-based similarity + decode. STUB."""

    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig, **kwargs):
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

    def infer(self, waveform, duration, tokens):
        raise NotImplementedError("SSL inference not yet implemented")

    def score(self, ctx, queries):
        raise NotImplementedError("SSL inference not yet implemented")

    def similarity(self, ctx):
        raise NotImplementedError("SSL inference not yet implemented")

    def decode(self, ctx, groups=None):
        raise NotImplementedError("SSL inference not yet implemented")
