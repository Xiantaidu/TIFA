import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor


def compute_determinacy(
    pred_spans: Tensor,
    similarity: Tensor,
    t_mask: Tensor,
    n_mask: Tensor,
    power: float = 2.0,
    width: int | None = 5,
) -> tuple[Tensor, Tensor]:
    """Per-item numerator and denominator for the PathDeterminacy metric.

    Args:
        pred_spans: ``[B, N_max, 2]`` predicted token spans in frames.
        similarity: ``[B, T_max, N_max]`` frame/token similarity matrix.
        t_mask: ``[B, T_max]`` valid frames.
        n_mask: ``[B, N_max]`` valid tokens.
        power: exponent for ReLU-activated similarities.
        width: neighborhood half-width in tokens (``None`` = unlimited).

    Returns:
        ``(numerator, denominator)`` as ``[B]`` tensors.
    """
    B, T_max, N_max = similarity.shape

    a = torch.relu(similarity) ** power
    a[~t_mask.unsqueeze(-1) | ~n_mask.unsqueeze(1)] = 0

    # Local neighborhood sum over token dim, or full sum when width is None
    if width is None:
        local_sum = a.sum(dim=-1, keepdim=True).expand(-1, -1, N_max)
    else:
        padded = F.pad(a, (width, width), value=0)
        windows = padded.unfold(2, 2 * width + 1, 1)
        local_sum = windows.sum(dim=-1)

    # Batched searchsorted needs padded onsets at inf
    onsets = pred_spans[:, :, 0].clone().to(dtype=torch.float32)
    onsets[~n_mask] = float("inf")
    offsets = pred_spans[:, :, 1].to(dtype=torch.float32)

    frame_idx = torch.arange(T_max, device=onsets.device).unsqueeze(0).expand(B, -1).contiguous()
    pos = torch.searchsorted(onsets, frame_idx, side="right") - 1

    # Token / gap classification
    pos_clamped = pos.clamp(min=0)
    is_token = (pos >= 0) & (frame_idx < offsets.gather(1, pos_clamped))

    regions = torch.where(is_token, pos_clamped + 1, torch.zeros_like(pos))

    N_per_item = n_mask.sum(dim=1).clamp(min=1) - 1
    gap_pos = (pos + 1).clamp(min=0)
    gap_pos = torch.min(gap_pos, N_per_item.unsqueeze(1))

    denom_idx = torch.where(is_token, pos_clamped, gap_pos)

    # Pad a zero column for gap gather (column 0 = 0 emission)
    a_padded = F.pad(a, (1, 0))

    num_contrib = a_padded.gather(2, regions.unsqueeze(-1)).squeeze(-1)
    num_contrib[~t_mask] = 0
    numerator = num_contrib.sum(dim=1)

    denom_contrib = local_sum.gather(2, denom_idx.unsqueeze(-1)).squeeze(-1)
    denom_contrib[~t_mask] = 0
    denominator = denom_contrib.sum(dim=1)

    return numerator, denominator


class PathDeterminacy(torchmetrics.Metric):
    """
    Reference-free metric that measures how strongly the model commits to
    its Viterbi-decoded path through the similarity matrix.

    For each frame assigned to a token, the ReLU-activated similarity to the
    assigned token is raised to ``power`` and accumulated in the numerator.
    The denominator accumulates the same activation for all tokens within
    ``width`` of the decoded token or gap position.  The ratio of the two
    sums is the per-frame average determinacy.

    Arguments:
        power: exponent for ReLU-activated similarities (default 2.0).
        width: neighborhood half-width in tokens for denominator competition
            (default 5).  ``None`` uses all tokens regardless of distance.

    Inputs:
        pred_spans  [B, N_max, 2]  --  predicted token spans in frames
        similarity  [B, T_max, N_max]  --  frame/token similarity matrix
        t_mask  [B, T_max]  --  valid frames
        n_mask  [B, N_max]  --  valid tokens

    Output:
        Scalar determinacy in [0, 1].
    """

    def __init__(self, power: float = 2.0, width: int | None = 5, **kwargs):
        super().__init__(**kwargs)
        self.power = power
        self.width = width
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_spans: Tensor,
        similarity: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> None:
        num, denom = compute_determinacy(
            pred_spans, similarity, t_mask, n_mask,
            power=self.power, width=self.width,
        )
        self.numerator += num.sum()
        self.denominator += denom.sum()

    def compute(self) -> Tensor:
        return self.numerator / (self.denominator + 1e-8)


def compute_confidence(
    pred_spans: Tensor,
    similarity: Tensor,
    t_mask: Tensor,
    n_mask: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Per-sample mean similarity within predicted spans.

    Args:
        pred_spans: ``[B, N_max, 2]`` predicted token spans in frames.
        similarity: ``[B, T_max, N_max]`` frame/token similarity matrix.
        t_mask: ``[B, T_max]`` valid frames.
        n_mask: ``[B, N_max]`` valid tokens.
        reduction: ``"mean"`` or ``"min"`` across tokens within each sample.

    Returns:
        ``[B]`` per-sample confidence values.
    """
    B, T_max, N_max = similarity.shape
    device = similarity.device

    t_idx = torch.arange(T_max, device=device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    onsets = pred_spans[:, :, 0].unsqueeze(1)   # [B, 1, N]
    offsets = pred_spans[:, :, 1].unsqueeze(1)  # [B, 1, N]

    span_mask = (t_idx >= onsets) & (t_idx < offsets)  # [B, T, N]
    span_mask = span_mask & t_mask.unsqueeze(-1) & n_mask.unsqueeze(1)

    token_sum = (similarity * span_mask.float()).sum(dim=1)  # [B, N]
    token_count = span_mask.float().sum(dim=1)                # [B, N]

    token_conf = torch.zeros(B, N_max, device=device)
    nonzero = token_count > 0
    token_conf[nonzero] = token_sum[nonzero] / token_count[nonzero]

    token_conf[~n_mask] = float("inf" if reduction == "min" else 0)

    if reduction == "min":
        result = token_conf.min(dim=1).values  # [B]
        result[result.isinf()] = 0.0
    else:
        valid_count = n_mask.sum(dim=1).clamp(min=1)
        result = token_conf.sum(dim=1) / valid_count

    return result


class Confidence(torchmetrics.Metric):
    """Reference-free metric: mean cosine similarity within each predicted span.

    For each token, the average similarity across its predicted duration is
    computed.  Zero-width spans get confidence 0.  The per-sample reduction
    (``"mean"`` or ``"min"``) is then averaged across items.

    Arguments:
        reduction: how to aggregate across tokens within a sample
            (``"mean"`` default, or ``"min"``).

    Inputs:
        pred_spans  [B, N_max, 2]  --  predicted token spans in frames
        similarity  [B, T_max, N_max]  --  frame/token similarity matrix
        t_mask  [B, T_max]  --  valid frames
        n_mask  [B, N_max]  --  valid tokens

    Output:
        Scalar confidence.
    """

    def __init__(self, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred_spans: Tensor,
        similarity: Tensor,
        t_mask: Tensor,
        n_mask: Tensor,
    ) -> None:
        values = compute_confidence(
            pred_spans, similarity, t_mask, n_mask,
            reduction=self.reduction,
        )
        self.total += values.sum()
        self.count += values.shape[0]

    def compute(self) -> Tensor:
        return self.total / self.count
