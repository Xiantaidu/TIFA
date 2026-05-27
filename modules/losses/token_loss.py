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

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, authentic: Tensor, mask_token: Tensor) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits.float(), authentic.float(), reduction='none'
        )
        return (loss * mask_token.float()).sum() / mask_token.float().sum().clamp(min=1)
