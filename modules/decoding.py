import numba
import numpy as np
import torch
from torch import Tensor

from modules.functional import cross_cosine_similarity


# ---------------------------------------------------------------------------
# Alignment decoding from cross-cosine-similarity
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _viterbi_decode_batch(
    sim: np.ndarray,    # [B, max_T, max_N] float32
    T_all: np.ndarray,  # [B] int64
    N_all: np.ndarray,  # [B] int64
    max_N: int,
) -> np.ndarray:        # [B, max_N, 2] int64
    """Viterbi decode with monotonicity constraint, batched over samples.

    States (per sample):
      G_i (0 <= i <= N): index i       - gap, next phoneme is i
      P_i (0 <= i < N):  index N+1+i   - inside phoneme i
    """
    B = sim.shape[0]
    spans_out = np.zeros((B, max_N, 2), dtype=np.int64)
    NEG_INF = np.float32(-1e9)

    for b in numba.prange(B):
        Ti = int(T_all[b])
        Ni = int(N_all[b])
        if Ni == 0:
            continue

        S = 2 * Ni + 1
        dp_prev = np.full(S, NEG_INF, dtype=np.float32)
        dp_cur = np.full(S, NEG_INF, dtype=np.float32)
        back = np.full((Ti, S), -1, dtype=np.int32)

        # t = 0
        dp_prev[0] = np.float32(0.0)  # G_0
        p0 = Ni + 1
        dp_prev[p0] = np.float32(sim[b, 0, 0])  # P_0 from G_0
        back[0, p0] = 0

        for t in range(1, Ti):
            dp_cur[:] = NEG_INF
            # G_0: from G_0 only (gap emission is 0)
            if dp_prev[0] > dp_cur[0]:
                dp_cur[0] = dp_prev[0]
                back[t, 0] = 0

            for i in range(Ni):
                p_idx = Ni + 1 + i  # P_i
                g_next = i + 1      # G_{i+1}
                emit = np.float32(sim[b, t, i])

                # P_i: stay in P_i
                if dp_prev[p_idx] > NEG_INF:
                    s = dp_prev[p_idx] + emit
                    if s > dp_cur[p_idx]:
                        dp_cur[p_idx] = s
                        back[t, p_idx] = p_idx
                # P_i: start from G_i
                if dp_prev[i] > NEG_INF:
                    s = dp_prev[i] + emit
                    if s > dp_cur[p_idx]:
                        dp_cur[p_idx] = s
                        back[t, p_idx] = i
                # P_i: advance from P_{i-1}
                if i > 0:
                    prev_p = Ni + 1 + (i - 1)
                    if dp_prev[prev_p] > NEG_INF:
                        s = dp_prev[prev_p] + emit
                        if s > dp_cur[p_idx]:
                            dp_cur[p_idx] = s
                            back[t, p_idx] = prev_p

                # G_{i+1}: stay in G_{i+1}
                if dp_prev[g_next] > NEG_INF:
                    if dp_prev[g_next] > dp_cur[g_next]:
                        dp_cur[g_next] = dp_prev[g_next]
                        back[t, g_next] = g_next
                # G_{i+1}: finish P_i
                if dp_prev[p_idx] > NEG_INF:
                    if dp_prev[p_idx] > dp_cur[g_next]:
                        dp_cur[g_next] = dp_prev[p_idx]
                        back[t, g_next] = p_idx

            dp_prev, dp_cur = dp_cur, dp_prev

        # Best end state at last frame
        best_s = np.argmax(dp_prev)
        # Backtrack
        states = np.empty(Ti, dtype=np.int32)
        s = best_s
        for t in range(Ti - 1, -1, -1):
            states[t] = s
            s = back[t, s]

        # Extract spans
        for i in range(Ni):
            p_idx = Ni + 1 + i
            first = -1
            last = -1
            for t in range(Ti):
                if states[t] == p_idx:
                    if first < 0:
                        first = t
                    last = t
            if first >= 0:
                spans_out[b, i, 0] = first
                spans_out[b, i, 1] = last + 1
            else:
                # Unreached token (T exhausted before N): place at last frame
                spans_out[b, i, 0] = Ti - 1
                spans_out[b, i, 1] = Ti - 1

    return spans_out


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

    T_all = frame_lengths.long().cpu().numpy()
    N_all = token_lengths.long().cpu().numpy()
    max_N = int(N_all.max())
    sim_np = sim.float().detach().cpu().numpy()

    spans_np = _viterbi_decode_batch(sim_np, T_all, N_all, max_N)

    return torch.from_numpy(spans_np).to(x_frame.device)
