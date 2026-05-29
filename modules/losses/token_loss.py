from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class TokenAuthenticityLoss(nn.Module):
    """
    Masked binary cross-entropy on per-token real/fake predictions.

    Inputs:
        logits: [B, N] raw logits per token
        authentic: [B, N] bool, True for original tokens, False for
            synthetically inserted or substituted tokens
        mask_token: [B, N] bool, non-padding tokens
    Returns:
        Scalar loss averaged over valid tokens.
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, authentic: Tensor, mask_token: Tensor) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), authentic.float(), reduction='none',
            pos_weight=self._pos_weight_tensor(logits),
        )
        return (loss * mask_token.float()).sum() / mask_token.float().sum().clamp(min=1)

    def _pos_weight_tensor(self, ref: Tensor) -> Tensor | None:
        if self.pos_weight is None:
            return None
        return ref.new_tensor(self.pos_weight)
