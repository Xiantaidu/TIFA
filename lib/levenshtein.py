"""Multi-string edit-path segmentation via Levenshtein alignment."""


def _levenshtein_align(a: list[str], b: list[str]) -> tuple[list, list]:
    """Pairwise Levenshtein alignment with backtracking.

    Returns ``(aligned_a, aligned_b)`` where each is a list of ``str | None``,
    with ``None`` marking positions where a gap was inserted into that sequence.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    aligned_a: list[str | None] = []
    aligned_b: list[str | None] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned_a.append(a[i - 1])
                aligned_b.append(b[j - 1])
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_a.append(a[i - 1])
            aligned_b.append(None)
            i -= 1
        else:
            aligned_a.append(None)
            aligned_b.append(b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return aligned_a, aligned_b


def _build_consensus(rows: list[list[str | None]]) -> list[str | None]:
    """Column-wise consensus from a row-oriented profile (all rows same length)."""
    n_cols = len(rows[0])
    consensus: list[str | None] = []
    for col_idx in range(n_cols):
        counts: dict[str, int] = {}
        for row in rows:
            t = row[col_idx]
            if t is not None:
                counts[t] = counts.get(t, 0) + 1
        consensus.append(max(counts, key=counts.get) if counts else None)
    return consensus


def _merge_profile(
    rows: list[list[str | None]], new_path: list[str]
) -> list[list[str | None]]:
    """Align *new_path* to the existing profile and merge it in."""
    consensus = _build_consensus(rows)
    al_c, al_p = _levenshtein_align(consensus, new_path)

    # Update existing rows: insert None where consensus had a gap
    new_rows: list[list[str | None]] = []
    for old_row in rows:
        new_row: list[str | None] = []
        old_idx = 0
        for c_tok in al_c:
            if c_tok is not None:
                new_row.append(old_row[old_idx])
                old_idx += 1
            else:
                new_row.append(None)
        new_rows.append(new_row)
    new_rows.append(al_p)
    return new_rows


def _segment_rows(rows: list[list[str | None]]) -> list[list[list[str]]]:
    """Segment aligned profile rows into shared/divergent segments."""
    if not rows:
        return []
    n_cols = len(rows[0])
    if n_cols == 0:
        return []

    col_match: list[bool] = []
    for col_idx in range(n_cols):
        tokens: set[str] = set()
        has_gap = False
        for row in rows:
            t = row[col_idx]
            if t is None:
                has_gap = True
            else:
                tokens.add(t)
        col_match.append(not has_gap and len(tokens) <= 1)

    segments: list[list[list[str]]] = []
    seg_start = 0
    for i in range(1, n_cols + 1):
        if i == n_cols or col_match[i] != col_match[seg_start]:
            seen: list[list[str]] = []
            for row in rows:
                sub = [t for t in row[seg_start:i] if t is not None]
                if sub not in seen:
                    seen.append(sub)
            segments.append(seen)
            seg_start = i

    return segments


def _align_multipath(paths: list[list[str]]) -> list[list[list[str]]]:
    """Align alternative sequences and segment into shared/divergent parts.

    Each input path is one alternative sequence.  Performs progressive
    multiple-sequence Levenshtein alignment, then segments the aligned
    columns into alternating *shared* and *divergent* segments.

    Returns a list of **segments**.  Each segment is a list of sub-paths
    (deduplicated).  A segment with a single sub-path means no alternatives
    at that position (a shared segment).

    >>> _align_multipath([["A", "B", "C", "D"], ["A", "E", "D"]])
    [[['A']], [['B', 'C'], ['E']], [['D']]]
    >>> _align_multipath([["l", "e"], ["l", "i", "ao"]])
    [[['l']], [['e'], ['i', 'ao']]]
    >>> _align_multipath([["h", "ao"]])
    [[['h', 'ao']]]
    """
    if not paths:
        return []
    if len(paths) == 1:
        return [[list(paths[0])]]

    # Progressive alignment — build profile row by row
    al_a, al_b = _levenshtein_align(paths[0], paths[1])
    rows: list[list[str | None]] = [
        list(al_a),
        list(al_b),
    ]

    for path in paths[2:]:
        rows = _merge_profile(rows, path)

    return _segment_rows(rows)


def _merge_shared(segments: list[list[list]]) -> list[list[list]]:
    """Merge consecutive segments that are both shared or both divergent.

    A segment with a single sub-path (width = 1) is a *shared* segment;
    a segment with multiple sub-paths (width > 1) is a *divergent* segment.

    Two consecutive shared segments are merged by concatenating their paths.
    Two consecutive divergent segments are merged via Cartesian product.
    A shared segment next to a divergent segment is left as-is.

    >>> _merge_shared([[['a']], [['b', 'c'], ['d']], [['e']]])
    [[['a']], [['b', 'c'], ['d']], [['e']]]
    >>> _merge_shared([[['a']], [['b']], [['c', 'd'], ['e']]])
    [[['a', 'b']], [['c', 'd'], ['e']]]
    >>> _merge_shared([[['B1'], ['B2']], [['C1'], ['C2']], [['D1']]])
    [[['B1', 'C1'], ['B1', 'C2'], ['B2', 'C1'], ['B2', 'C2']], [['D1']]]
    """
    if not segments:
        return []
    merged = [segments[0]]
    for sub_paths in segments[1:]:
        prev = merged[-1]
        if len(sub_paths) == 1 and len(prev) == 1:
            prev[0].extend(sub_paths[0])
        elif len(sub_paths) > 1 and len(prev) > 1:
            merged[-1] = [p + s for p in prev for s in sub_paths]
        else:
            merged.append(sub_paths)
    return merged


def segment_groups(groups: list[list[list]]) -> list[list[list]]:
    """Align and merge alternative paths from multiple consecutive groups.

    Calls :func:`_align_multipath` on each group, then :func:`_merge_shared`
    across all resulting segments.

    Each element of *groups* is a set of alternative paths, where each path
    is a sequence of elements.  Returns the final merged segment list.

    >>> segment_groups([[['a']], [['b1'], ['b2']], [['c']]])
    [[['a']], [['b1'], ['b2']], [['c']]]
    >>> segment_groups([[['a']], [['b']], [['c']]])
    [[['a', 'b', 'c']]]
    """
    all_segments: list[list[list]] = []
    for group_paths in groups:
        all_segments.extend(_align_multipath(group_paths))
    return _merge_shared(all_segments)
