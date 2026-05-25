"""Tensorized path sampling for text-only alignment.

All operations are fully batched — no Python loops.

Two sampling strategies:

- :func:`sample_paths_uniform`: each of *k* paths independently draws a random
  alt per segment.  Samples uniformly from the full product space.

- :func:`sample_paths_perm`: batched randperm via argsort.  Every alt of
  every segment appears at least once.

- :func:`extract_tokens`: gathers token sequences via the segments bridge,
  then compacts interspersed zeros to the right using a stable argsort
  re-mapping.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and compact token sequences for sampled paths.

    Args:
        paths: ``[B, N_max, W_max]`` token ids (0 = padding).
        segments: ``[B, N_max]`` 1-based segment indices (0 = padding).
                  Padding positions are identified by ``segments == 0``.
        choices: ``[B, K, S_max]`` from any of the sampling functions above.

    Returns:
        ``(compacted, lengths)`` where *compacted* is ``[B, K, max_len]``
        with no trailing zeros and *lengths* is ``[B, K]``.
    """
    B_val, N_max, _W_max = paths.shape
    K = choices.shape[1]

    # seg_ids [B, N_max] — 0-based; padding → 0 after clamp
    seg_ids = segments.clamp(min=1) - 1

    # alt per (item, path, grid_pos)
    alt_per_pos = torch.gather(
        choices, 2, seg_ids.unsqueeze(1).expand(-1, K, -1)
    )  # [B, K, N_max]

    # gather tokens
    b_idx = torch.arange(B_val).view(B_val, 1, 1).expand(-1, K, N_max)
    n_idx = torch.arange(N_max).view(1, 1, N_max).expand(B_val, K, -1)
    tokens = paths[b_idx, n_idx, alt_per_pos]  # [B, K, N_max]

    # zero out batch-padding positions
    tokens[(segments == 0).unsqueeze(1).expand(-1, K, -1)] = 0

    # compact interspersed zeros to the right via stable sort
    mask = tokens != 0
    priority = torch.where(
        mask,
        torch.arange(N_max, device=tokens.device).view(1, 1, N_max),
        N_max,
    )
    perm = priority.argsort(dim=-1, stable=True)
    compacted = tokens.gather(-1, perm)

    lengths = mask.sum(dim=-1)  # [B, K]
    max_len = lengths.max()
    compacted = compacted[:, :, :max_len]

    return compacted, lengths
