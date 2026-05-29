import numba
import numpy as np
import torch
from torch import Tensor

from modules.functional import cross_cosine_similarity


# ---------------------------------------------------------------------------
# Alignment decoding from cross-cosine-similarity
# ---------------------------------------------------------------------------


@numba.njit
def _viterbi_decode(
    sim: np.ndarray,  # [T, N] float32
    T: int,
    N: int,
) -> np.ndarray:      # [N, 2] int64
    """Viterbi decode with monotonicity constraint.

    States:
      G_i (0 <= i <= N): index i       - gap, next phoneme is i
      P_i (0 <= i < N):  index N+1+i   - inside phoneme i
    """
    if N == 0:
        return np.zeros((0, 2), dtype=np.int64)

    S = 2 * N + 1  # total states
    NEG_INF = np.float32(-1e9)

    dp_prev = np.full(S, NEG_INF, dtype=np.float32)
    dp_cur = np.full(S, NEG_INF, dtype=np.float32)
    back = np.full((T, S), -1, dtype=np.int32)

    # t = 0
    dp_prev[0] = np.float32(0.0)  # G_0
    p0 = N + 1
    dp_prev[p0] = sim[0, 0]  # P_0 from G_0
    back[0, p0] = 0

    for t in range(1, T):
        dp_cur[:] = NEG_INF
        # G_0: from G_0 only
        # (gap emission is 0)
        if dp_prev[0] > dp_cur[0]:
            dp_cur[0] = dp_prev[0]
            back[t, 0] = 0

        for i in range(N):
            p_idx = N + 1 + i  # P_i
            g_next = i + 1     # G_{i+1}

            # P_i: from P_i (stay), G_i (start), or P_{i-1} (advance)
            emit = np.float32(sim[t, i])

            # Stay in P_i
            if dp_prev[p_idx] > NEG_INF:
                s = dp_prev[p_idx] + emit
                if s > dp_cur[p_idx]:
                    dp_cur[p_idx] = s
                    back[t, p_idx] = p_idx
            # Start from G_i
            if dp_prev[i] > NEG_INF:
                s = dp_prev[i] + emit
                if s > dp_cur[p_idx]:
                    dp_cur[p_idx] = s
                    back[t, p_idx] = i
            # Advance from P_{i-1}
            if i > 0:
                prev_p = N + 1 + (i - 1)
                if dp_prev[prev_p] > NEG_INF:
                    s = dp_prev[prev_p] + emit
                    if s > dp_cur[p_idx]:
                        dp_cur[p_idx] = s
                        back[t, p_idx] = prev_p

            # G_{i+1}: from G_{i+1} (stay) or P_i (finish)
            if dp_prev[g_next] > NEG_INF:
                if dp_prev[g_next] > dp_cur[g_next]:
                    dp_cur[g_next] = dp_prev[g_next]
                    back[t, g_next] = g_next
            if dp_prev[p_idx] > NEG_INF:
                if dp_prev[p_idx] > dp_cur[g_next]:
                    dp_cur[g_next] = dp_prev[p_idx]
                    back[t, g_next] = p_idx

        dp_prev, dp_cur = dp_cur, dp_prev

    # Best end state at last frame
    best_s = np.argmax(dp_prev)
    # Backtrack
    states = np.empty(T, dtype=np.int32)
    s = best_s
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = back[t, s]

    # Extract spans
    spans = np.zeros((N, 2), dtype=np.int64)
    for i in range(N):
        p_idx = N + 1 + i
        frames = np.where(states == p_idx)[0]
        if len(frames) > 0:
            spans[i, 0] = frames[0]
            spans[i, 1] = frames[-1] + 1
        else:
            # Unreached token (T exhausted before N): place at last frame
            spans[i, 0] = T - 1
            spans[i, 1] = T - 1
    return spans


def decode_alignment(
    x_frame: Tensor,        # [B, T, C]
    x_token: Tensor,        # [B, N, C]
    frame_lengths: Tensor,  # [B] int64, valid frames per sample (trailing padding)
    token_lengths: Tensor,  # [B] int64, valid tokens per sample (trailing padding)
    temperature: float = 0.1,
) -> Tensor:               # [B, N, 2] int64, (onset, offset)
    """Viterbi-decode frame/token features into per-token spans.

    Uses a monotonicity-constrained state space (gap + phoneme states) to
    produce spans that respect left-to-right token ordering.

    Args:
        x_frame: [B, T, C] frame features from backbone (trailing-padded).
        x_token: [B, N, C] token embeddings (trailing-padded).
        frame_lengths: [B] int64, number of valid frames per sample.
        token_lengths: [B] int64, number of valid tokens per sample.
        temperature: cosine similarity temperature (should match training).

    Returns:
        spans [B, N, 2] int64, (onset, offset) in frames.  Padded entries
        are (0, 0).
    """
    sim = cross_cosine_similarity(x_frame, x_token, temperature)  # [B, T, N]

    T_all = frame_lengths.long().cpu()
    N_all = token_lengths.long().cpu()
    max_N = int(N_all.max().item())
    sim_np = sim.float().detach().cpu().numpy()

    spans_list = []
    for i in range(sim.shape[0]):
        Ti = int(T_all[i].item())
        Ni = int(N_all[i].item())

        spans_i = _viterbi_decode(np.ascontiguousarray(sim_np[i, :Ti, :Ni]), Ti, Ni)

        if Ni < max_N:
            padded = np.zeros((max_N, 2), dtype=np.int64)
            padded[:Ni] = spans_i
            spans_i = padded

        spans_list.append(torch.from_numpy(spans_i).to(x_frame.device))

    return torch.stack(spans_list, dim=0)
