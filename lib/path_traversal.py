"""Tensorized operations on path grids for forced-alignment inference.

A **path grid** is a ``[B, N_max, W_max]`` tensor produced by the dataset
where each column *w* holds one alternative token sequence for a segment.
``segments`` ``[B, N_max]`` maps each grid row to its segment (1-based,
0 = padding), and ``widths`` ``[B, S_max]`` gives the number of
alternatives per segment.

**Extract**

- :func:`extract_tokens` -- gather token sequences for chosen alternatives.
- :func:`compact_sequences` -- left-pack non-zero tokens.

**Sample**

- :func:`sample_paths_perm` -- batched randperm covering every alt at
  least once.
- :func:`sample_paths_uniform` -- independent random alt per segment.

All operations are fully batched -- no Python loops over *B*.
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
        *args: torch.Tensor,
        segments: torch.Tensor,
        choices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Extract token sequences for sampled paths (un-compacted).

    Args:
        *args: one or more source tensors ``[B, N_max, W_max]``
            (0 = padding), e.g. *paths* and *words*.
        segments: ``[B, N_max]`` 1-based segment indices (0 = padding).
        choices: ``[B, ..., S_max]`` from any sampling function.

    Returns:
        Tuple of tensors ``[B, ..., N_max]``, one per source, matching
        the dims of *choices* and 0-padded.
    """
    B_val, N_max, W_max = args[0].shape
    mid_dims = choices.shape[1:-1]
    choices_2d = choices.reshape(choices.shape[0], -1, choices.shape[-1])  # [B, K, S_max]
    K = choices_2d.shape[1]

    # Prepend column of zeros so 1-based segments index directly (padding→0 → alt 0)
    choices_0 = torch.cat([
        torch.zeros(B_val, K, 1, dtype=choices_2d.dtype, device=choices_2d.device),
        choices_2d,
    ], dim=2)  # [B, K, 1 + S_max]

    alt_per_pos = torch.gather(
        choices_0, 2, segments.unsqueeze(1).expand(-1, K, -1),
    )  # [B, K, N_max]

    b_idx = torch.arange(B_val).view(B_val, 1, 1).expand(-1, K, N_max)
    n_idx = torch.arange(N_max).view(1, 1, N_max).expand(B_val, K, -1)
    pad_mask = (segments == 0).unsqueeze(1).expand(-1, K, -1)  # [B, K, N_max]

    out_shape = (B_val, *mid_dims, N_max)
    results: list[torch.Tensor] = []
    for src in args:
        tokens = src[b_idx, n_idx, alt_per_pos]  # [B, K, N_max]
        tokens[pad_mask] = 0
        results.append(tokens.reshape(out_shape))
    return tuple(results)


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
        ``(compacted_tokens, *compacted_args)`` -- ``[..., M]``
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
