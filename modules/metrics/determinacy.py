import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor


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
        B, T_max, N_max = similarity.shape

        a = torch.relu(similarity) ** self.power
        a[~t_mask.unsqueeze(-1) | ~n_mask.unsqueeze(1)] = 0

        # Local neighborhood sum over token dim, or full sum when width is None
        if self.width is None:
            local_sum = a.sum(dim=-1, keepdim=True).expand(-1, -1, N_max)
        else:
            padded = F.pad(a, (self.width, self.width), value=0)
            windows = padded.unfold(2, 2 * self.width + 1, 1)
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
        self.numerator += num_contrib.sum()

        denom_contrib = local_sum.gather(2, denom_idx.unsqueeze(-1)).squeeze(-1)
        denom_contrib[~t_mask] = 0
        self.denominator += denom_contrib.sum()

    def compute(self) -> Tensor:
        return self.numerator / (self.denominator + 1e-8)
