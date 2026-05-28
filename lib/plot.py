from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def spectrogram_to_figure(spectrogram, title=None):
    fig = plt.figure(figsize=(12, 3))
    plt.pcolormesh(spectrogram.T, vmin=-14, vmax=4)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def cross_similarity_to_figure(sim, regions=None, title=None) -> "plt.Figure":
    """Plot cross cosine similarity matrix [N, T].

    sim: [N, T] float array, values in [-1, 1]
    regions: [T] int array, 1-based region index (0=gap), optional GT overlay
    title: optional title string
    """
    N, T = sim.shape
    fig_width = max(12, min(T / 60, 20))
    fig_height = max(4, min(N / 4, 10))
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.pcolormesh(sim, vmin=-1, vmax=1, cmap="RdBu_r", zorder=1)
    plt.colorbar(label="cosine similarity")

    if regions is not None:
        # Shade gap frames (regions == 0) with gray vertical bands
        is_gap = (regions == 0)
        if is_gap.any():
            edges = np.diff(np.concatenate([[False], is_gap, [False]]).astype(int))
            for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                plt.axvspan(start, end, color="gray", alpha=0.25, zorder=2)

        # Region boundary lines
        changes = np.where(regions[:-1] != regions[1:])[0]
        for t in changes:
            plt.axvline(x=t + 1, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    plt.xlabel("Frame")
    plt.ylabel("Token")
    plt.xlim(0, T)
    plt.ylim(0, N)
    if title is not None:
        plt.title(title, fontsize=12)
    plt.tight_layout()
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
            ax.text(x, c + max_count * 0.01, str(c), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(symbols, fontsize=7)
        ax.set_xlim(-0.6, len(symbols) - 0.4)
        ax.set_title(f"{lang}  ({len(symbols)} symbols, {sum(counts)} occurrences)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max_count * 1.15)
    fig.tight_layout()
    return fig
