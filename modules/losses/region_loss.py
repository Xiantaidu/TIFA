import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


def cross_cosine_similarity(x: Tensor, y: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Compute cross-sequence cosine similarity matrix.
    x: [..., T, C]  (e.g. frame features)
    y: [..., N, C]  (e.g. token embeddings)
    Returns: [..., T, N]
    """
    x_norm = F.normalize(x.float(), p=2, dim=-1, eps=1e-8)
    y_norm = F.normalize(y.float(), p=2, dim=-1, eps=1e-8)
    return (x_norm @ y_norm.transpose(-1, -2)) / temperature


class FrameAlignmentLoss(nn.Module):
    """
    Per-frame cross-entropy on the cross-modal similarity matrix [B, T, N].
    Every non-gap frame classifies which of the N phoneme tokens it belongs to.

    Inputs:
        x_frame: [B, T, C] frame features from backbone
        x_token: [B, N, C] token embeddings
        regions: [B, T] 1-based region index (0 = gap between phonemes)
        mask_frame: [B, T] bool, non-padding frames
        mask_token: [B, N] bool, non-padding tokens
    Returns:
        Scalar loss.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        x_frame: Tensor,
        x_token: Tensor,
        regions: Tensor,
        mask_frame: Tensor,
        mask_token: Tensor,
    ) -> Tensor:
        B, T, N = x_token.shape[0], x_frame.shape[1], x_token.shape[1]
        valid_frame = mask_frame & (regions > 0)  # [B, T]
        if valid_frame.sum() == 0:
            return x_frame.new_zeros(())

        sim = cross_cosine_similarity(x_frame, x_token, self.temperature)  # [B, T, N]

        # Mask invalid tokens as keys, but only for frames that participate
        sim = sim.masked_fill(
            ~mask_token.unsqueeze(1) & valid_frame.unsqueeze(2), float('-inf')
        )

        target = regions - 1  # [B, T], gap → -1
        loss = F.cross_entropy(
            sim.reshape(B * T, N),
            target.reshape(B * T),
            ignore_index=-1,
            reduction='mean',
        )
        return loss


class SpanContrastiveLoss(nn.Module):
    """
    Bidirectional InfoNCE between span-pooled frame features and token embeddings.

    Pools frame features within each phoneme's [start, end) span, then computes
    InfoNCE contrastive loss in both frame→token and token→frame directions.

    Inputs:
        x_frame: [B, T, C] frame features from backbone
        x_token: [B, N, C] token embeddings
        spans: [B, N, 2] (inclusive_start, exclusive_end) per token
        mask_frame: [B, T] bool, non-padding frames
        mask_token: [B, N] bool, non-padding tokens
    Returns:
        Scalar loss. Returns 0 if fewer than 2 valid tokens.
    """

    def __init__(self, temperature: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(
        self,
        x_frame: Tensor,
        x_token: Tensor,
        spans: Tensor,
        mask_frame: Tensor,
        mask_token: Tensor,
    ) -> Tensor:
        T = x_frame.shape[1]
        device = x_frame.device

        if mask_token.sum() < 2:
            return x_frame.new_zeros(())

        # Pool frame features within each span
        t_idx = torch.arange(T, device=device)
        in_span = (t_idx >= spans[..., 0:1]) & (t_idx < spans[..., 1:2])  # [B, N, T]
        in_span = in_span & mask_frame.unsqueeze(1)
        counts = in_span.float().sum(dim=-1, keepdim=True).clamp(min=1)
        x_pooled = (in_span.float() @ x_frame) / counts  # [B, N, C]

        S = cross_cosine_similarity(x_pooled, x_token, self.temperature)  # [B, N, N]

        # Mask invalid positions in both dimensions
        S = S.masked_fill(~mask_token.unsqueeze(1), float('-inf'))  # invalid keys
        S = S.masked_fill(~mask_token.unsqueeze(2), float('-inf'))  # invalid queries

        # F→T: each span picks its own token
        log_prob_f2t = F.log_softmax(S, dim=-1)
        diag_f2t = log_prob_f2t.diagonal(dim1=-2, dim2=-1)
        loss_f2t = -diag_f2t[mask_token].mean()

        if not self.bidirectional:
            return loss_f2t

        # T→F: each token picks its own span
        log_prob_t2f = F.log_softmax(S.transpose(-1, -2), dim=-1)
        diag_t2f = log_prob_t2f.diagonal(dim1=-2, dim2=-1)
        loss_t2f = -diag_t2f[mask_token].mean()

        return (loss_f2t + loss_t2f) / 2
