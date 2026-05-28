import torch
import torch.nn as nn
import torch.nn.functional as F


def hmm_forward_loss_with_emission(
    emission: torch.Tensor,
    frame_lens: torch.Tensor,
    token_lens: torch.Tensor,
) -> torch.Tensor:
    """CTC-based HMM forward loss from pre-computed emission matrix.

    Disables blank (global -inf) and uses targets [0, 1, ..., N-1],
    forcing monotonic left-to-right alignment through token states.
    """
    B, T_max, N_max = emission.shape

    # mask padding token states
    state_mask = torch.arange(N_max, device=emission.device).unsqueeze(0) >= token_lens.unsqueeze(1)
    emission.masked_fill_(state_mask.unsqueeze(1), -1e9)

    # append blank column (set to -inf, effectively disabled)
    blank_col = torch.full((B, T_max, 1), -1e9, device=emission.device)
    logits = torch.cat([emission, blank_col], dim=-1)  # [B, T_max, N_max+1]

    log_probs = F.log_softmax(logits, dim=-1)

    targets = torch.arange(N_max, device=emission.device).unsqueeze(0).expand(B, -1)

    loss = F.ctc_loss(
        log_probs.transpose(0, 1),  # [T_max, B, N_max+1]
        targets,
        input_lengths=frame_lens,
        target_lengths=token_lens,
        blank=N_max,  # global blank index
        reduction='mean',
        zero_infinity=True,
    )
    return loss


class HMMForwardLoss(nn.Module):
    """HMM forward loss via CTC with disabled blank.

    Computes cosine-similarity emission from frame and token features,
    then runs forced-alignment CTC with targets [0, 1, ..., N-1].
    """

    def forward(
        self,
        x_features: torch.Tensor,
        token_features: torch.Tensor,
        frame_lens: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor:
        emission = torch.bmm(
            F.normalize(x_features, dim=-1),
            F.normalize(token_features, dim=-1).transpose(1, 2),
        )  # [B, T_max, N_max]
        return hmm_forward_loss_with_emission(emission, frame_lens, token_lens)
