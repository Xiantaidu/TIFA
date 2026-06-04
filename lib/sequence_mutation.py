"""Random sequence edits for supervised forced-alignment training.

Applies random insertions, deletions and substitutions to a phoneme-timing
sample, then aligns original tokens to real target tokens via Levenshtein DP.
The left-to-right optimal alignment naturally prioritizes the first occurrence
of consecutive identical tokens.
"""

import random

import torch
from torch import Tensor

from lib.levenshtein import align_sequences

__all__ = [
    "apply_sequence_edits",
    "apply_mask_mutations",
]


def _rebuild_regions(spans: torch.Tensor, T: int) -> torch.Tensor:
    """Rebuild per-frame region indices from spans via scatter-add + cumsum."""
    M = spans.shape[0]
    device = spans.device
    delta = torch.zeros(T + 1, dtype=torch.long, device=device)
    ids = torch.arange(1, M + 1, dtype=torch.long, device=device)
    delta.scatter_add_(0, spans[:, 0], ids)
    delta.scatter_add_(0, spans[:, 1], -ids)
    return delta.cumsum(0)[:T]


def apply_sequence_edits(
    *,
    tokens: Tensor,
    spans: Tensor,
    regions: Tensor,
    min_token: int,
    max_token: int,
    p_sub: float,
    p_del: float,
    p_ins: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply random edits to a token sequence.

    Args:
        tokens: ``[N]`` int64, 1-based vocabulary group IDs.
        spans: ``[N, 2]`` int64, (inclusive_start, exclusive_end) frames.
        regions: ``[T]`` int64, per-frame 1-based token index, 0 for stop.
        min_token: smallest valid token ID for insertions/substitutions (inclusive).
        max_token: largest valid token ID for insertions/substitutions (inclusive).
        p_sub: per-token substitution probability.
        p_del: per-token deletion probability.
        p_ins: per-gap insertion probability.

    Returns:
        ``(new_tokens, new_spans, new_regions, original_tokens)`` where:
        - new_tokens ``[N']`` int64, edited token IDs.
        - new_spans ``[N', 2]`` int64, (inclusive_start, exclusive_end) frames.
        - new_regions ``[T]`` int64, per-frame 1-based token index, 0 for stop.
        - original_tokens ``[N']`` int64, the phoneme ID that belongs at each
          position: the original token ID for kept and substituted tokens, 0
          for insertions.
    """
    N = tokens.shape[0]
    T = regions.shape[0]
    device = tokens.device

    # ------------------------------------------------------------------
    # Phase 1  --  Sample edits and build target sequence
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
    tgt_original_tokens: list[int] = []

    for i in range(N):
        if do_ins[i]:
            tgt_tokens.append(random.randint(min_token, max_token))
            tgt_original_tokens.append(0)

        if do_del[i]:
            continue

        if do_sub[i]:
            tgt_tokens.append(random.randint(min_token, max_token))
            tgt_original_tokens.append(orig_tokens[i])
        else:
            tgt_tokens.append(orig_tokens[i])
            tgt_original_tokens.append(orig_tokens[i])

    if do_ins[N]:
        tgt_tokens.append(random.randint(min_token, max_token))
        tgt_original_tokens.append(0)

    M = len(tgt_tokens)

    # ------------------------------------------------------------------
    # Phase 2  --  Levenshtein DP alignment
    # ------------------------------------------------------------------
    # Only allow matching kept tokens (not substitutions or insertions).
    # Wrap as (token, is_kept) so the cost function can distinguish.
    tgt_wrapped = [
        (t, tgt_original_tokens[j] == t)
        for j, t in enumerate(tgt_tokens)
    ]
    orig_wrapped = [(t, True) for t in orig_tokens]

    def _edit_cost(a: tuple[int, bool], b: tuple[int, bool]) -> int | None:
        if a[0] == b[0] and b[1]:
            return 0
        return None

    al_orig, al_mut = align_sequences(
        orig_wrapped, tgt_wrapped, match_cost=_edit_cost, prefer_indel=True,
    )

    # ------------------------------------------------------------------
    # Phase 3  --  Assign spans and rebuild regions
    # ------------------------------------------------------------------

    orig_spans = spans.tolist()
    new_spans_list: list[tuple[int, int]] = []
    orig_ptr = 0
    for j in range(len(al_mut)):
        # Advance orig_ptr for deleted original tokens (gap in mut)
        if al_mut[j] is None:
            if al_orig[j] is not None:
                orig_ptr += 1
            continue
        if al_orig[j] is None:
            new_spans_list.append((0, 0))
        else:
            new_spans_list.append(tuple(orig_spans[orig_ptr]))
            orig_ptr += 1

    new_tokens = torch.tensor(tgt_tokens, dtype=torch.long, device=device)
    new_spans = torch.tensor(new_spans_list, dtype=torch.long, device=device)
    original_tokens = torch.tensor(tgt_original_tokens, dtype=torch.long, device=device)
    new_regions = _rebuild_regions(new_spans, T)
    return new_tokens, new_spans, new_regions, original_tokens


def apply_mask_mutations(
        *,
        tokens: torch.Tensor,
        spans: torch.Tensor,
        regions: torch.Tensor,
        p_mask: float,
        p_insert: float,
        p_chain: float,
        max_chain: int,
        mask_token: int,
        space_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MASK replacement and chained insertion, resolve with alignment.

    MASK replacement randomly replaces existing tokens with *mask_token*.
    Chained insertion randomly inserts MASK runs at gaps with geometric
    continuation.  After both mutations, Levenshtein alignment with
    MASK-as-wildcard resolves which MASKs correspond to original tokens
    and which are genuine insertions.

    The alignment step is necessary because a MASK can arise from either
    a replacement or an insertion, and the model sees only the final token
    sequence -- without alignment the target for each MASK position would
    be ambiguous.  Treating MASK as a wildcard during alignment ensures
    each replacement-MASK is deterministically paired with the leftmost
    available original token, while unpaired MASKs are labeled as SPACE.

    Args:
        tokens: ``[N]`` int64, original token IDs (no MASK).
        spans: ``[N, 2]`` int64, (start, end) frames.
        regions: ``[T]`` int64, per-frame 1-based token indices.
        p_mask: per-token replacement probability.
        p_insert: per-gap probability of starting an insertion run.
        p_chain: probability of extending a run by one more MASK.
        max_chain: maximum number of consecutive MASKs in a run.
        mask_token: token ID used for MASK.
        space_token: token ID used for SPACE (insertion target).

    Returns:
        ``(new_tokens, new_spans, new_regions, targets)`` where targets
        is *space_token* for inserted positions and the original token ID
        for kept or replaced positions.
    """

    N = tokens.shape[0]
    T = regions.shape[0]
    device = tokens.device

    orig_list = tokens.tolist()
    span_list = spans.tolist()

    # Step 1: MASK replacement (per-token)
    mut_list: list[int] = []
    for i in range(N):
        if random.random() < p_mask:
            mut_list.append(mask_token)
        else:
            mut_list.append(orig_list[i])

    # Step 2: MASK chained insertion (per-gap, no blocking needed --
    # the alignment step resolves any ambiguity)
    insertions: list[list[int]] = [[] for _ in range(N + 1)]
    for gap in range(N + 1):
        if random.random() >= p_insert:
            continue
        run: list[int] = [mask_token]
        while len(run) < max_chain and random.random() < p_chain:
            run.append(mask_token)
        insertions[gap] = run

    # Build pre-alignment sequence
    pre_tok: list[int] = []
    for i in range(N):
        pre_tok.extend(insertions[i])
        pre_tok.append(mut_list[i])
    pre_tok.extend(insertions[N])

    # Step 3: Levenshtein alignment (MASK wildcard: exact=0, wildcard=1)
    def _mask_cost(a: int, b: int) -> int | None:
        if a == b:
            return 0
        if b == mask_token:
            return 1
        return None

    al_orig, al_mut = align_sequences(
        orig_list, pre_tok, match_cost=_mask_cost,
        prefer_indel=True, prefer_insert=True,
    )

    # Step 4: Build targets, tokens, and spans from alignment.
    # A gap in al_orig (None) means this position is an insertion.
    # A gap in al_mut (None) means the original token was deleted.
    new_tok: list[int] = []
    new_spans: list[tuple[int, int]] = []
    targets: list[int] = []
    orig_ptr = 0
    for j in range(len(al_mut)):
        if al_mut[j] is None:
            continue  # deletion -- skip this position
        new_tok.append(al_mut[j])
        if al_orig[j] is None:
            new_spans.append((0, 0))
            targets.append(space_token)
        else:
            new_spans.append(tuple(span_list[orig_ptr]))
            targets.append(orig_list[orig_ptr])
            orig_ptr += 1

    new_tokens = torch.tensor(new_tok, dtype=torch.long, device=device)
    new_spans_t = torch.tensor(new_spans, dtype=torch.long, device=device)
    new_targets = torch.tensor(targets, dtype=torch.long, device=device)
    new_regions = _rebuild_regions(new_spans_t, T)
    return new_tokens, new_spans_t, new_regions, new_targets
