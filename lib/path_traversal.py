"""Tensorized operations on path grids for forced-alignment inference.

A **path grid** is a ``[B, N_max, W_max]`` tensor produced by the dataset
where each column *w* holds one alternative token sequence for a segment.
``segments`` ``[B, N_max]`` maps each grid row to its segment (1-based,
0 = padding), and ``widths`` ``[B, S_max]`` gives the number of
alternatives per segment.

The core inference pipeline operates on a given path grid:

1. **Sample** -- draw alt choices per segment from the grid.

   - :func:`sample_paths_perm` -- batched randperm covering every alt
     at least once.
   - :func:`sample_paths_uniform` -- independent random alt per segment.
   - :func:`build_length_grid` -- group alternatives by subpath length,
     producing a coarser grid for more efficient sampling.

2. **Extract** -- gather token sequences for the sampled choices.

   - :func:`extract_tokens` -- gather from the grid via alt indices.
   - :func:`compact_sequences` -- left-pack non-zero tokens.

3. **Score** -- compare candidate tokens against model predictions
   (model forward lives in :mod:`inference.backend`).  These helpers
   build the inputs:

   - :func:`layout_segments` -- per-segment lengths and start positions
     in a concatenated token sequence.
   - :func:`segment_grid_starts` -- first grid-row per (item, segment).
   - :func:`scatter_real_tokens` -- overwrite non-divergent positions
     in a MASK_TOKEN-filled tensor with real tokens from the grid.
   - :func:`gather_query_tokens` -- gather subpath tokens into query
     rows for batched MLM scoring.

4. **Accumulate** -- aggregate per-query scores to pick the best
   alternative per segment.

   - :func:`accumulate_best_alts` -- scatter-add into a
     ``[B, S_max, W_max]`` table and argmax.

**Utilities**

- :func:`expand_by_lengths` -- expand N groups by their lengths into
  flat ``(group_idx, offset)`` index arrays.

All operations are fully batched -- no Python loops over *B*, *U*, or *Q*.
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

    seg_ids = segments.clamp(min=1) - 1  # 0-based; padding -> 0

    alt_per_pos = torch.gather(
        choices_2d, 2, seg_ids.unsqueeze(1).expand(-1, K, -1),
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


def build_length_grid(
        paths: torch.Tensor,
        segments: torch.Tensor,
        widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group subpaths by distinct length within each segment.

    Args:
        paths: ``[B, N_max, W_max]`` token IDs, 0-padded.
        segments: ``[B, N_max]`` 1-based segment indices, 0 = padding.
        widths: ``[B, S_max]`` number of alternatives per segment.

    Returns:
        length_widths: ``[B, S_max]`` int64, number of distinct lengths
            per segment (0 for padding segments).
        length_values: ``[B, S_max, L_max]`` int64, actual token-count
            for each length alternative, 0-padded.  Sorted ascending.
        lengths: ``[B, S_max, W_max]`` int64, subpath length of each
            original alt (0 for invalid alts).
    """
    B, N_max, W_max = paths.shape
    S_max = widths.shape[1]
    device = paths.device

    # Per-alt lengths via scatter_add over the segment dimension.
    # segments uses 0 for padding and 1..S_max for real segments, so it
    # already doubles as the scatter index (slot 0 = padding, ignored).
    token_valid = paths != 0  # [B, N_max, W_max]
    seg_expanded = segments.unsqueeze(-1).expand(-1, -1, W_max)  # [B, N_max, W_max]

    lengths_with_pad = torch.zeros(B, S_max + 1, W_max, dtype=torch.int64, device=device)
    lengths_with_pad.scatter_add_(1, seg_expanded, token_valid.long())
    lengths = lengths_with_pad[:, 1:, :]  # [B, S_max, W_max]

    alt_mask = torch.arange(W_max, device=device).view(1, 1, W_max) < widths.unsqueeze(-1)
    lengths[~alt_mask] = 0

    # Sort to find unique lengths per segment
    INF = torch.iinfo(torch.int64).max
    sort_key = torch.where(alt_mask, lengths, INF)  # [B, S_max, W_max]
    sorted_keys, _sort_indices = sort_key.sort(dim=-1)  # [B, S_max, W_max]

    is_new = torch.ones(B, S_max, W_max, dtype=torch.bool, device=device)
    is_new[:, :, 1:] = sorted_keys[:, :, 1:] != sorted_keys[:, :, :-1]
    is_new = is_new & (sorted_keys != INF)

    length_widths = is_new.sum(dim=-1)  # [B, S_max]

    # Pack distinct lengths into [B, S_max, L_max]
    L_max = int(length_widths.max().item())
    if L_max == 0:
        length_values = torch.zeros(B, S_max, 1, dtype=torch.int64, device=device)
        return length_widths, length_values, lengths

    unique_idx = is_new.int().cumsum(dim=-1)  # [B, S_max, W_max], values in [0, L_max]
    unique_idx[~is_new] = 0  # non-unique positions -> slot 0 (discarded by [:, :, 1:])

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, S_max, W_max)
    s_idx = torch.arange(S_max, device=device).view(1, S_max, 1).expand(B, -1, W_max)

    length_values = torch.zeros(B, S_max, L_max + 1, dtype=torch.int64, device=device)
    # Scatter sorted_keys into length_values using unique_idx as the column index.
    # Each (b, s, unique_idx_val) receives at most one value (for is_new positions),
    # so advanced-index assignment is safe.
    length_values[b_idx, s_idx, unique_idx] = sorted_keys
    length_values = length_values[:, :, 1:]  # [B, S_max, L_max]

    return length_widths, length_values, lengths


def expand_by_lengths(lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand N groups by their lengths into flat index arrays.

    Args:
        lengths: ``[N]`` int64, number of elements in each group.

    Returns:
        group_idx: ``[total]`` int64, which group (0..N-1) each element
            belongs to.
        offset: ``[total]`` int64, position within that group.

    Example:
        lengths = [2, 0, 3] -> group_idx = [0,0,2,2,2], offset = [0,1,0,1,2]
    """
    N = lengths.numel()
    total = int(lengths.sum().item())
    device = lengths.device
    group_idx = torch.repeat_interleave(
        torch.arange(N, device=device), lengths,
    )
    offset = torch.arange(total, device=device) - torch.repeat_interleave(
        torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            lengths[:-1].cumsum(dim=0),
        ]), lengths,
    )
    return group_idx, offset


def segment_grid_starts(segments: torch.Tensor, S_max: int) -> torch.Tensor:
    """First grid-row index for each segment in each item.

    Args:
        segments: ``[B, N_max]`` int64, 1-based segment IDs (0 = padding).
        S_max: total number of segments.

    Returns:
        ``[B, S_max]`` int64, first row index (0 if segment is absent).
    """
    device = segments.device
    seg_eq = segments.unsqueeze(-1) == torch.arange(
        1, S_max + 1, device=device,
    ).view(1, 1, -1)
    return seg_eq.int().argmax(dim=1)


def scatter_real_tokens(
        target: torch.Tensor,
        paths: torch.Tensor,
        seg_starts: torch.Tensor,
        seg_lens: torch.Tensor,
        non_div_mask: torch.Tensor,
        item_idx: torch.Tensor,
        grid_starts: torch.Tensor,
) -> None:
    """Overwrite non-divergent segment positions in *target* with real tokens.

    *target* should already be filled with MASK_TOKEN at valid positions
    and 0 at padding positions.

    Args:
        target: ``[U, N]`` int64, modified in-place.
        paths: ``[B, N_grid, W_max]`` int64, path grid.
        seg_starts: ``[U, S_max]`` int64, start position per segment.
        seg_lens: ``[U, S_max]`` int64, length per segment.
        non_div_mask: ``[U, S_max]`` bool, True for non-divergent segments.
        item_idx: ``[U]`` int64, maps each unique choice to its item in *paths*.
        grid_starts: ``[B, S_max]`` int64, first grid row per (item, segment).
    """
    nd_active = non_div_mask & (seg_lens > 0)
    nd_u, nd_s = nd_active.nonzero(as_tuple=True)
    nd_item = item_idx[nd_u]
    nd_start = seg_starts[nd_u, nd_s]
    nd_len = seg_lens[nd_u, nd_s]

    nd_exp, nd_j = expand_by_lengths(nd_len)

    nd_src_item = nd_item[nd_exp]
    nd_src_seg = nd_s[nd_exp]
    nd_src_row = grid_starts[nd_src_item, nd_src_seg] + nd_j
    target[nd_u[nd_exp], nd_start[nd_exp] + nd_j] = (
        paths[nd_src_item, nd_src_row, 0]
    )


def gather_query_tokens(
        U: int, Q: int, N: int,
        paths: torch.Tensor,
        active_u: torch.Tensor,
        active_s: torch.Tensor,
        active_w: torch.Tensor,
        active_item: torch.Tensor,
        q_start: torch.Tensor,
        q_len: torch.Tensor,
        q_local: torch.Tensor,
        grid_starts: torch.Tensor,
) -> torch.Tensor:
    """Gather subpath tokens from the path grid into query rows ``[U, Q, N]``.

    Each query row ``(active_u[i], q_local[i])`` contains the subpath tokens
    for alternative *active_w[i]* of segment *active_s[i]*, placed at
    positions ``[q_start[i], q_start[i]+q_len[i])``, with 0 elsewhere.

    Args:
        U, Q, N: dimensions of the output tensor.
        paths: ``[B, N_grid, W_max]`` int64, path grid.
        active_u, active_s, active_w, active_item: ``[Q]`` int64.
        q_start, q_len: ``[Q]`` int64, segment position and length.
        q_local: ``[Q]`` int64, local query index within each u.
        grid_starts: ``[B, S_max]`` int64, first grid row per (item, segment).

    Returns:
        ``[U, Q, N]`` int64, 0-padded query tensor.
    """
    device = paths.device
    expanded, j_idx = expand_by_lengths(q_len)

    src_item = active_item[expanded]
    src_seg = active_s[expanded]
    src_alt = active_w[expanded]
    src_row = grid_starts[src_item, src_seg] + j_idx
    src_tokens = paths[src_item, src_row, src_alt]

    queries = torch.zeros(U, Q, N, dtype=torch.int64, device=device)
    queries[
        active_u[expanded], q_local[expanded],
        q_start[expanded] + j_idx,
    ] = src_tokens
    return queries


def layout_segments(
        non_div: torch.Tensor,
        base_lens: torch.Tensor,
        length_values: torch.Tensor,
        unique_lchoices: torch.Tensor,
        sample_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute segment lengths and start positions for each unique choice.

    Args:
        non_div: ``[U, S_max]`` bool, True for non-divergent segments.
        base_lens: ``[B, S_max]`` int64, length of alt 0 per (item, segment).
        length_values: ``[B, S_max, L_max]`` int64, distinct lengths per segment.
        unique_lchoices: ``[U, S_max]`` int64, which length alternative per segment.
        sample_idx: ``[U]`` int64, maps each unique choice to its item.

    Returns:
        seg_lens: ``[U, S_max]`` int64, token count per segment.
        seg_starts: ``[U, S_max]`` int64, start position in the concatenated sequence.
        N_max: int, maximum total sequence length across all choices.
    """
    device = non_div.device
    U, S_max = non_div.shape
    s_idx = torch.arange(S_max, device=device).unsqueeze(0).expand(U, -1)
    b_idx = sample_idx.unsqueeze(-1).expand(-1, S_max)
    div_lens = length_values[b_idx, s_idx, unique_lchoices]  # [U, S_max]
    seg_lens = torch.where(non_div, base_lens[sample_idx], div_lens)  # [U, S_max]

    seg_starts = torch.zeros(U, S_max, dtype=torch.int64, device=device)
    seg_starts[:, 1:] = seg_lens[:, :-1].cumsum(dim=-1)
    N_max = int(seg_lens.sum(dim=-1).max().item())
    return seg_lens, seg_starts, N_max


def accumulate_best_alts(
        q_scores: torch.Tensor,
        q_u: torch.Tensor,
        q_local: torch.Tensor,
        q_item: torch.Tensor,
        q_seg: torch.Tensor,
        q_alt: torch.Tensor,
        B: int,
        S_max: int,
        W_max: int,
) -> torch.Tensor:
    """Scatter query scores into [B, S_max, W_max] and argmax.

    Args:
        q_scores: ``[U, Q]`` float, mean score per query row.
        q_u, q_local, q_item, q_seg, q_alt: ``[Q]`` int64 index tensors.
        B, S_max, W_max: dimensions of the output table.

    Returns:
        ``[B, S_max]`` int64, best alternative index per (item, segment).
    """
    device = q_scores.device
    Q = q_u.numel()
    flat_idx = q_item * (S_max * W_max) + q_seg * W_max + q_alt
    flat_scores = q_scores[q_u, q_local]

    acc_flat = torch.zeros(B * S_max * W_max, device=device)
    acc_flat.scatter_add_(0, flat_idx, flat_scores)
    cnt_flat = torch.zeros(B * S_max * W_max, device=device)
    cnt_flat.scatter_add_(0, flat_idx, torch.ones(Q, device=device))
    acc = acc_flat.reshape(B, S_max, W_max)
    cnt = cnt_flat.reshape(B, S_max, W_max)
    return (acc / cnt.clamp(min=1)).argmax(dim=-1)
