import torch
import torch.nn as nn
import torch.nn.functional as F






def hmm_forward_loss(audio_out, text_expanded, mel_lens:torch.Tensor, text_lens:torch.Tensor):
    B, T_max, D = audio_out.shape
    S_max = text_expanded.shape[1]

    emission = torch.bmm(
        F.normalize(audio_out, dim=-1),
        F.normalize(text_expanded, dim=-1).transpose(1, 2)
    )  # [B, T_max, S_max]

    # mask padding 子状态
    state_mask = torch.arange(S_max, device=emission.device).unsqueeze(0) >= text_lens.unsqueeze(1)
    emission.masked_fill_(state_mask.unsqueeze(1), -1e9)  # [B, T_max, S_max]

    # 添加 blank 列（也设为 -inf，禁用 blank）
    blank_col = torch.full((B, T_max, 1), -1e9, device=emission.device)
    logits = torch.cat([emission, blank_col], dim=-1)  # [B, T_max, S_max+1]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, T_max, S_max+1]

    # targets
    targets = torch.arange(S_max, device=emission.device).unsqueeze(0).expand(B, -1)

    loss = F.ctc_loss(
        log_probs.transpose(0, 1),  # [T_max, B, S_max+1]
        targets,
        input_lengths=mel_lens,
        target_lengths=text_lens,
        blank=S_max,  # 全局 blank index
        reduction='mean',
        zero_infinity=True
    )

    return loss

def hmm_forward_loss_with_emission(emission, mel_lens:torch.Tensor, text_lens:torch.Tensor):
    # B, T_max, D = audio_out.shape
    # S_max = text_expanded.shape[1]
    #
    # emission = torch.bmm(
    #     F.normalize(audio_out, dim=-1),
    #     F.normalize(text_expanded, dim=-1).transpose(1, 2)
    # )  # [B, T_max, S_max]
    B, T_max, S_max = emission.shape
    # mask padding 子状态
    state_mask = torch.arange(S_max, device=emission.device).unsqueeze(0) >= text_lens.unsqueeze(1)
    emission.masked_fill_(state_mask.unsqueeze(1), -1e9)  # [B, T_max, S_max]

    # 添加 blank 列（也设为 -inf，禁用 blank）
    blank_col = torch.full((B, T_max, 1), -1e9, device=emission.device)
    logits = torch.cat([emission, blank_col], dim=-1)  # [B, T_max, S_max+1]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, T_max, S_max+1]

    # targets
    targets = torch.arange(S_max, device=emission.device).unsqueeze(0).expand(B, -1)

    loss = F.ctc_loss(
        log_probs.transpose(0, 1),  # [T_max, B, S_max+1]
        targets,
        input_lengths=mel_lens,
        target_lengths=text_lens,
        blank=S_max,  # 全局 blank index
        reduction='mean',
        zero_infinity=True
    )

    return loss