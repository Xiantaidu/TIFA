import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpectrogramReconstructionLoss(nn.Module):
    """L1 or L2 loss between reconstructed and original spectrogram.

    Only valid (non-padding) frames contribute to the loss.

    Inputs:
        x_recon: [B, T, C] reconstructed spectrogram
        target: [B, T, C] original clean spectrogram
        t_mask: [B, T] bool, non-padding frames
    Returns:
        Scalar loss.
    """

    def __init__(self, loss_type: str = "L1"):
        super().__init__()
        if loss_type not in ("L1", "L2"):
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self,
        x_recon: Tensor,
        target: Tensor,
        t_mask: Tensor,
    ) -> Tensor:
        if self.loss_type == "L1":
            per_element = F.l1_loss(x_recon, target, reduction="none")
        else:
            per_element = F.mse_loss(x_recon, target, reduction="none")

        per_frame = per_element.mean(dim=-1)
        valid = per_frame.masked_select(t_mask)
        if valid.numel() == 0:
            return torch.zeros(1, device=x_recon.device, requires_grad=True)
        return valid.mean()
