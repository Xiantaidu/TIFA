"""Tensorized path sampling for text-only alignment.

All operations are fully batched — no Python loops over items, paths, or
segments (except :func:`sample_paths_uniform` with ``unique=True``, which
loops over the batch dimension for per-item deduplication).

Three sampling strategies:

- :func:`sample_paths_uniform`: each of *k* paths independently draws a random
  alt per segment.  Samples uniformly from the full product space.

- :func:`sample_paths_randperm`: batched randperm via argsort.  Each alt of
  every segment appears exactly once if unique = True.

- :func:`sample_paths_enumerate`: deterministic cyclic enumeration repeated
  *r* times over the largest width.  O(r · max(widths)).

- :func:`extract_tokens`: gathers token sequences via the segments bridge,
  then compacts interspersed zeros to the right using a stable argsort
  re-mapping.
"""
import torch


def sample_paths_uniform(widths: torch.Tensor, k: int, unique: bool = True) -> torch.Tensor:
    """Sample up to *k* complete paths uniformly from the product space.

    Each path draws an independent random alt per segment:
    ``choices[b, s, p] = floor(rand * widths[b, p])``.

    Args:
        widths: ``[B, S_max]`` padded with **1** (not 0).
        k: max number of paths to sample per item.
        unique: if True, deduplicate per item and pad to the batch-wide
                max unique count.  This introduces a Python loop over B
                (not trace-friendly; B ≈ few dozen is acceptable).

    Returns:
        choices ``[B, K, S_max]`` where *K* is ``k`` when ``unique=False``,
        otherwise the maximum unique path count across all items.
    """
    B, S_max = widths.shape
    device = widths.device
    choices = (torch.rand(B, S_max, k, device=device) * widths.unsqueeze(-1)).long()
    choices = choices.clamp(max=(widths.unsqueeze(-1) - 1).clamp(min=0))
    choices = choices.permute(0, 2, 1)  # [B, k, S_max]

    if not unique:
        return choices

    # Deduplicate per item; pad to max unique count.
    # Per-slice unique is not available in PyTorch — encoding item identity
    # into the rows and flattening is no simpler at these sizes.  Loop over B
    # is acceptable since B ≈ few dozen, but also means this op is NOT
    # trace-friendly (variable output size per slice).  If you ever wrap this
    # with torch.compile / torch.export, pass unique=False and dedup outside
    # the graph.
    uniq_list = [torch.unique(choices[b], dim=0) for b in range(B)]
    K = max(u.shape[0] for u in uniq_list)
    out = torch.zeros(B, K, S_max, dtype=torch.long, device=device)
    for b, u in enumerate(uniq_list):
        Ku = u.shape[0]
        out[b, :Ku] = u
        if Ku < K:
            out[b, Ku:] = u[-1:]  # repeat last path
    return out


def sample_paths_randperm(widths: torch.Tensor) -> torch.Tensor:
    """Enumerate one path per alternative via batched randperm (coverage-based).

    Each segment independently draws a random permutation of its alternatives.
    Returns ``max(widths)`` paths, guaranteeing every alt of every segment
    appears at least once.  Entries beyond a segment's width are garbage —
    the caller must clamp with ``widths``.

    Args:
        widths: ``[B, S_max]`` padded with **1** (not 0).

    Returns:
        choices ``[B, W_max, S_max]`` where ``W_max = max(widths)``.
    """
    B, S_max = widths.shape
    W_max = widths.max()
    device = widths.device

    rand = torch.rand(B, S_max, W_max, device=device)
    valid = torch.arange(W_max, device=device).view(1, 1, W_max) < widths.unsqueeze(-1)
    rand[~valid] = float("inf")

    perm = rand.argsort(dim=-1)  # [B, S_max, W_max]
    return perm.permute(0, 2, 1)  # [B, W_max, S_max]


def sample_paths_enumerate(widths: torch.Tensor, r: int = 1) -> torch.Tensor:
    """Deterministic cyclic enumeration repeated *r* times.

    Cycles through alternatives in order ``[0, 1, 2, 0, 1, 2, …]`` across
    ``r * W_max`` paths where ``W_max = max(widths)``.  Entries beyond a
    segment's width are garbage — the caller must clamp with ``widths``.

    Args:
        widths: ``[B, S_max]`` padded with **1** (not 0).
        r: number of full cycles over the largest width.

    Returns:
        choices ``[B, r * W_max, S_max]``.
    """
    B, S_max = widths.shape
    W_max = widths.max()
    device = widths.device
    cycle = torch.arange(W_max, device=device).repeat(r)  # [r * W_max]
    return cycle.view(1, -1, 1).expand(B, -1, S_max)  # [B, r * W_max, S_max]


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
        choices: ``[B, K, S_max]`` from :func:`sample_paths_uniform` or
                :func:`sample_paths_randperm`.

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
