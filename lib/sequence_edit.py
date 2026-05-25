"""Random sequence edits for supervised forced-alignment training.

Applies random insertions, deletions and substitutions to a phoneme-timing
sample, then aligns original tokens to real target tokens via Levenshtein DP.
The left-to-right optimal alignment naturally prioritizes the first occurrence
of consecutive identical tokens.
"""

import random

import torch
from torch import Tensor

__all__ = ["apply_sequence_edits"]


def apply_sequence_edits(
    *,
    tokens: Tensor,
    spans: Tensor,
    regions: Tensor,
    vocab_size: int,
    p_sub: float,
    p_del: float,
    p_ins: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply random edits to a token sequence.

    Args:
        tokens: ``[N]`` int64, 1-based vocabulary group IDs.
        spans: ``[N, 2]`` int64, (inclusive_start, exclusive_end) frames.
        regions: ``[T]`` int64, per-frame 1-based token index, 0 for stop.
        vocab_size: number of token IDs (including padding 0).
        p_sub: per-token substitution probability.
        p_del: per-token deletion probability.
        p_ins: per-gap insertion probability.

    Returns:
        ``(new_tokens, new_spans, new_regions, fake)`` where:
        - new_tokens ``[N']``, new_spans ``[N', 2]``, new_regions ``[T]``,
          fake ``[N']`` bool.
    """
    N = tokens.shape[0]
    T = regions.shape[0]
    device = tokens.device

    # ------------------------------------------------------------------
    # Phase 1 — Sample edits and build target sequence
    # ------------------------------------------------------------------

    do_sub = [random.random() < p_sub for _ in range(N)]
    do_del = [random.random() < p_del for _ in range(N)]
    do_ins = [random.random() < p_ins for _ in range(N + 1)]

    # Guard: can't delete all tokens
    if all(do_del):
        keep_idx = random.randrange(N)
        do_del[keep_idx] = False

    # Priority: sub > del > ins (at most one operation per place)
    for i in range(N):
        if do_sub[i]:
            do_del[i] = False
        if do_sub[i] or do_del[i]:
            if i + 1 <= N:
                do_ins[i + 1] = False

    orig_tokens = tokens.tolist()
    tgt_tokens: list[int] = []
    tgt_fake: list[bool] = []

    for i in range(N):
        if do_ins[i]:
            tgt_tokens.append(random.randint(1, vocab_size - 1))
            tgt_fake.append(True)

        if do_del[i]:
            continue

        if do_sub[i]:
            tgt_tokens.append(random.randint(1, vocab_size - 1))
            tgt_fake.append(True)
        else:
            tgt_tokens.append(orig_tokens[i])
            tgt_fake.append(False)

    if do_ins[N]:
        tgt_tokens.append(random.randint(1, vocab_size - 1))
        tgt_fake.append(True)

    M = len(tgt_tokens)

    # ------------------------------------------------------------------
    # Phase 2 — Levenshtein DP alignment
    # ------------------------------------------------------------------

    INF = N + M + 1

    dp = [[0] * (M + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        dp[i][0] = i
    for j in range(M + 1):
        dp[0][j] = j

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if orig_tokens[i - 1] == tgt_tokens[j - 1] and not tgt_fake[j - 1]:
                match = dp[i - 1][j - 1]
            else:
                match = INF
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            dp[i][j] = min(match, delete, insert)

    # Backtrack — prefer delete over match when both are optimal, so that
    # real target tokens match the *earliest* available original position
    # (leftmost-match = first-of-consecutive-identicals prior).
    alignment: list[int | None] = [None] * M
    i, j = N, M
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
        else:
            alignment[j - 1] = i - 1
            i -= 1
            j -= 1

    # ------------------------------------------------------------------
    # Phase 3 — Assign spans and rebuild regions
    # ------------------------------------------------------------------

    orig_spans = spans.tolist()
    new_spans_list: list[tuple[int, int]] = []
    for j in range(M):
        orig_idx = alignment[j]
        if orig_idx is not None:
            new_spans_list.append(tuple(orig_spans[orig_idx]))
        else:
            new_spans_list.append((0, 0))

    new_tokens = torch.tensor(tgt_tokens, dtype=torch.long, device=device)
    new_spans = torch.tensor(new_spans_list, dtype=torch.long, device=device)
    fake = torch.tensor(tgt_fake, dtype=torch.bool, device=device)

    delta = torch.zeros(T + 1, dtype=torch.long, device=device)
    ids = torch.arange(1, M + 1, dtype=torch.long, device=device)
    delta.scatter_add_(0, new_spans[:, 0], ids)
    delta.scatter_add_(0, new_spans[:, 1], -ids)
    new_regions = delta.cumsum(0)[:T]

    return new_tokens, new_spans, new_regions, fake
