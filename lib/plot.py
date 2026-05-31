from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def cross_similarity_to_figure(sim, regions=None, title=None, token_labels=None) -> "plt.Figure":
    """Plot cross cosine similarity matrix [N, T].

    sim: [N, T] float array, values in [-1, 1]
    regions: [T] int array, 1-based region index (0=gap), optional GT overlay
    title: optional title string
    token_labels: optional list of N strings for y-axis tick labels
    """
    N, T = sim.shape
    fig_width = max(12, min(T / 60, 20))
    fig_height = max(4, min(N / 4, 10))
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.pcolormesh(sim, vmin=-1, vmax=1, cmap="RdBu_r", zorder=1)
    plt.colorbar().set_label("cosine similarity", fontsize=10)

    if regions is not None:
        # Split region boundaries into phone-phone and phone-gap
        changes = np.where(regions[:-1] != regions[1:])[0]
        is_phone_phone = (regions[changes] > 0) & (regions[changes + 1] > 0)
        pp_changes = changes[is_phone_phone] + 1
        pg_changes = changes[~is_phone_phone] + 1

        # Phone-gap boundaries: solid lines
        plt.vlines(pg_changes, 0, N, colors="black", linestyles="-", linewidth=1.0,
                   label="GT phone-gap")
        # Phone-phone boundaries: dashed lines
        plt.vlines(pp_changes, 0, N, colors="black", linestyles="--", linewidth=1.0,
                   label="GT phone-phone")
        plt.legend(loc="lower right", fontsize=10)

    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Token", fontsize=12)
    if token_labels is not None:
        centers = [i + 0.5 for i in range(N)]
        plt.yticks(centers, token_labels[:N], fontsize=10)
    plt.xlim(0, T)
    plt.ylim(0, N)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def alignment_to_figure(
        spectrogram,
        token_labels,
        pred_spans,
        gt_spans=None,
        title=None,
        stagger: int = 10,
) -> "plt.Figure":
    """Plot spectrogram with alignment span boundaries overlaid.

    GT spans (if provided) are shown in the upper half of the
    spectrogram; predicted spans in the lower half.  When GT is
    absent, predicted spans use the full height.

    Phone-phone boundaries (dashed) and phone-gap boundaries (solid)
    are distinguished.  Text labels are staggered across *stagger*
    vertical levels to reduce overlap.

    spectrogram: [T, n_bins] float array
    token_labels: list of N strings
    pred_spans: [N, 2] int array, (onset, offset) in frames
    gt_spans: optional [N, 2] int array
    title: optional title string
    stagger: number of vertical text levels to cycle through
    """

    def _classify(spans):
        onsets = set()
        offsets = set()
        for j in range(N):
            o, e = int(spans[j, 0]), int(spans[j, 1])
            if o < e:
                onsets.add(o)
                offsets.add(e)
        pp = sorted(onsets & offsets)
        pg = sorted((onsets | offsets) - (onsets & offsets))
        return pp, pg

    T, n_bins = spectrogram.shape
    N = len(token_labels)

    fig_width = max(12, min(T / 40, 24))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.pcolormesh(
        np.arange(T + 1), np.arange(n_bins + 1),
        spectrogram.T, vmin=-14, vmax=4,
        zorder=1, rasterized=True,
    )

    def _draw_alignment(spans, ymin, ymax, color, prefix):
        pp, pg = _classify(spans)
        height = (ymax - ymin) / stagger
        for j in range(N):
            onset, offset = int(spans[j, 0]), int(spans[j, 1])
            if onset < offset:
                level = j % stagger
                y = ymin + (level + 0.5) * height
                ax.text(
                    (onset + offset) / 2, y, token_labels[j],
                    ha="center", va="center", fontsize=10, color="white", zorder=3,
                )
        if pg:
            ax.vlines(pg, ymin, ymax,
                      colors=color, linestyles="-", linewidth=1.0, zorder=2,
                      label=f"{prefix} phone-gap")
        if pp:
            ax.vlines(pp, ymin, ymax,
                      colors=color, linestyles="--", linewidth=1.0, zorder=2,
                      label=f"{prefix} phone-phone")

    has_gt = gt_spans is not None

    if has_gt:
        gt_ymin, gt_ymax = n_bins / 2, n_bins
        pred_ymin, pred_ymax = 0, n_bins / 2
    else:
        gt_ymin, gt_ymax = None, None
        pred_ymin, pred_ymax = 0, n_bins

    if has_gt:
        _draw_alignment(gt_spans, gt_ymin, gt_ymax, "gold", "GT")

    _draw_alignment(pred_spans, pred_ymin, pred_ymax, "red", "pred")

    ax.legend(loc="lower right", fontsize=10)

    ax.set_xlim(0, T)
    ax.set_ylim(0, n_bins)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Bins", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=15)

    fig.tight_layout()
    return fig


def topk_bar_figure(
        labels: list[str],
        values: list[float],
        title: str,
        reverse: bool = True,
) -> "plt.Figure":
    """Horizontal bar chart of label -> value.

    The caller is responsible for formatting labels (e.g. resolving token
    IDs to human-readable symbols) and for any truncation.
    By default, the largest value is at the top; pass *reverse=False*
    when lower values are worse.
    """
    pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=reverse)
    sorted_labels = [p[0] for p in pairs]
    sorted_values = [p[1] for p in pairs]

    n = len(sorted_labels)
    fig_height = max(4, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ys = range(n)
    ax.barh(ys, sorted_values, align="center")
    ax.set_yticks(ys)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.invert_yaxis()  # worst at top
    ax.set_title(title, fontsize=15)
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(sorted_values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=10)

    fig.tight_layout()
    return fig


def vocab_distribution_to_figure(symbol_counts: dict[str, int]) -> "plt.Figure":
    lang_data: dict[str, dict[str, int]] = defaultdict(dict)
    for sym, count in symbol_counts.items():
        lang = sym.split("/")[0] if "/" in sym else "_"
        short = sym.split("/")[-1] if "/" in sym else sym
        lang_data[lang][short] = lang_data[lang].get(short, 0) + count

    sorted_langs = sorted(lang_data, key=lambda ln: (ln != "_", ln))
    n_langs = len(sorted_langs)
    if n_langs == 0:
        return None

    max_symbols = max(len(lang_data[ln]) for ln in sorted_langs)
    fig_width = max(16, max_symbols * 0.4)
    fig, axes = plt.subplots(n_langs, 1, figsize=(fig_width, 5 * n_langs), squeeze=False)
    for i, lang in enumerate(sorted_langs):
        ax = axes[i][0]
        data = lang_data[lang]
        symbols = sorted(data.keys())
        xs = range(len(symbols))
        counts = [data[s] for s in symbols]
        ax.bar(xs, counts)
        max_count = max(counts)
        for x, c in zip(xs, counts):
            ax.text(x, c + max_count * 0.01, str(c), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(symbols, fontsize=10)
        ax.set_xlim(-0.6, len(symbols) - 0.4)
        ax.set_title(f"{lang}  ({len(symbols)} symbols, {sum(counts)} occurrences)", fontsize=15)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max_count * 1.15)
    fig.tight_layout()
    return fig
