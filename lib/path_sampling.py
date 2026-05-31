"""Tensorized path sampling for text-only alignment.

All operations are fully batched  --  no Python loops.

Two sampling strategies:

- :func:`sample_paths_uniform`: each of *k* paths independently draws a random
  alt per segment.  Samples uniformly from the full product space.

- :func:`sample_paths_perm`: batched randperm via argsort.  Every alt of
  every segment appears at least once.

- :func:`extract_tokens`: gathers token sequences via the segments bridge
  (un-compacted).

- :func:`compact_sequences`: compacts interspersed zeros to the right via
  stable argsort across the last dim, truncating to the max non-zero count.
"""
import torch


def sample_paths_uniform(widths: torch.Tensor, k: int) -> torch.Tensor:
    """Sample *k* complete paths uniformly from the product space.

    Each path draws an independent random alt per segment:
    ``choices[b, s, p] = floor(rand * widths[b, p])``.

    Args:
        widths: ``[B, S_max]`` padded with **1** (not 0).
        k: number of paths to sample per item.

    Returns:
        choices ``[B, k, S_max]``.
    """
    B, S_max = widths.shape
    device = widths.device
    choices = (torch.rand(B, S_max, k, device=device) * widths.unsqueeze(-1)).long()
    choices = choices.clamp(max=widths.unsqueeze(-1) - 1)  # guard rand*w rounding to w
    return choices.permute(0, 2, 1)  # [B, k, S_max]


def sample_paths_perm(widths: torch.Tensor, r: int = 1) -> torch.Tensor:
    """Coverage-based sampling via batched randperm, repeated *r* times.

    Each repeat is an independent random permutation, so increasing *r*
    explores more cross-combinations.  Every alt of every segment appears
    at least once.

    Args:
        widths: ``[B, S_max]`` padded with **1** (not 0).
        r: number of independent permutations.

    Returns:
        choices ``[B, r * W_max, S_max]`` where ``W_max = max(widths)``.
    """
    B, S_max = widths.shape
    W_max = widths.max()
    device = widths.device

    rand = torch.rand(B, S_max, W_max, r, device=device)  # [B, S_max, W_max, r]
    valid = torch.arange(W_max, device=device).view(1, 1, W_max, 1) < widths.view(B, S_max, 1, 1)
    rand[~valid.expand(-1, -1, -1, r)] = float("inf")

    perm = rand.argsort(dim=2)  # [B, S_max, W_max, r]
    return (perm % widths.view(B, S_max, 1, 1)).permute(0, 3, 2, 1).reshape(B, r * W_max, S_max)


def extract_tokens(
        paths: torch.Tensor,
        segments: torch.Tensor,
        choices: torch.Tensor,
) -> torch.Tensor:
    """Extract token sequences for sampled paths (un-compacted).

    Args:
        paths: ``[B, N_max, W_max]`` token ids (0 = padding).
        segments: ``[B, N_max]`` 1-based segment indices (0 = padding).
        choices: ``[B, ..., S_max]`` from any sampling function.

    Returns:
        tokens ``[B, ..., N_max]`` matching the dims of *choices*,
        0-padded.
    """
    B_val, N_max, _W_max = paths.shape
    mid_dims = choices.shape[1:-1]  # dimensions between B and S_max
    choices = choices.reshape(choices.shape[0], -1, choices.shape[-1])  # [B, K, S_max]
    K = choices.shape[1]

    seg_ids = segments.clamp(min=1) - 1  # 0-based; padding -> 0

    alt_per_pos = torch.gather(
        choices, 2, seg_ids.unsqueeze(1).expand(-1, K, -1)
    )  # [B, K, N_max]

    b_idx = torch.arange(B_val).view(B_val, 1, 1).expand(-1, K, N_max)
    n_idx = torch.arange(N_max).view(1, 1, N_max).expand(B_val, K, -1)
    tokens = paths[b_idx, n_idx, alt_per_pos]  # [B, K, N_max]

    # zero out batch-padding positions
    tokens[(segments == 0).unsqueeze(1).expand(-1, K, -1)] = 0

    out_shape = (B_val, *mid_dims, N_max)
    return tokens.reshape(out_shape)


def compact_sequences(
        tokens: torch.Tensor, *args: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Compact sequences via stable sort on the token mask.

    All inputs must be 0-padded and have the same shape.  The first
    argument (tokens) determines the compaction mask.

    Args:
        tokens: ``[..., N]``, 0 = padding.
        *args: ``[..., N]``, same shape as tokens, 0-padded.

    Returns:
        ``(compacted_tokens, *compacted_args)`` — ``[..., M]``
        where ``M = mask.sum(dim=-1).max()``.
    """
    mask = tokens != 0
    N_max = tokens.shape[-1]

    priority = torch.where(
        mask,
        torch.arange(N_max, device=tokens.device).expand_as(tokens),
        N_max,
    )
    perm = priority.argsort(dim=-1, stable=True)
    compacted_tokens = tokens.gather(-1, perm)

    compacted_args: list[torch.Tensor] = []
    for arg in args:
        arg = arg.clone()
        arg[~mask] = 0
        compacted_args.append(arg.gather(-1, perm))

    max_len = mask.sum(dim=-1).max()
    compacted_tokens = compacted_tokens[..., :max_len]
    compacted_args = [a[..., :max_len] for a in compacted_args]

    return compacted_tokens, *compacted_args
