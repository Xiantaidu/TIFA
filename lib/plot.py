from collections import defaultdict

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np


def spectrogram_to_figure(spectrogram, title=None):
    fig = plt.figure(figsize=(12, 3))
    plt.pcolormesh(spectrogram.T, vmin=-14, vmax=4)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def similarity_to_figure(similarities, durations, title=None):
    dur_cumsum = np.cumsum(durations)
    fig = plt.figure(figsize=(9, 9))
    plt.pcolormesh(similarities, vmin=-1, vmax=1)
    for i in range(durations.shape[0]):
        rect = matplotlib.patches.Rectangle(
            xy=(dur_cumsum[i] - durations[i], dur_cumsum[i] - durations[i]),
            width=durations[i], height=durations[i],
            edgecolor="red", fill=False, linewidth=1.5,
        )
        plt.gca().add_patch(rect)
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def boundary_to_figure(
        boundaries_gt: np.ndarray, boundaries_pred: np.ndarray,
        threshold: float = None,
        boundaries_tp: np.ndarray = None,
        boundaries_fp: np.ndarray = None,
        boundaries_fn: np.ndarray = None,
        title=None
):
    figure_width = 12
    figure_height = 6
    fig = plt.figure(figsize=(figure_width, figure_height))
    plt.plot(boundaries_gt, color="b", label="gt")
    plt.plot(boundaries_pred, color="r", label="pred")
    if threshold is not None:
        plt.plot([0, boundaries_gt.shape[0]], [threshold, threshold], color="black", linestyle="--")
    positions = np.arange(boundaries_gt.shape[0], dtype=np.int64)
    circle_radius = 10
    x_min = 0
    x_max = boundaries_gt.shape[0]
    y_min = 0
    y_max = 1.1
    ratio = (figure_width / figure_height) * (y_max - y_min) / (x_max - x_min)

    def _draw_circles(x_index, y_arr, color, label):
        label_added = False
        for pos in positions[x_index]:
            plt.gca().add_patch(
                matplotlib.patches.Ellipse(
                    xy=(pos, y_arr[pos]),
                    width=circle_radius, height=circle_radius * ratio,
                    edgecolor=color, fill=False,
                    linewidth=1.5, label=(label if not label_added else None)
                )
            )
            label_added = True

    if boundaries_tp is not None:
        _draw_circles(positions[boundaries_tp], boundaries_pred, "green", "match")
    if boundaries_fp is not None:
        _draw_circles(positions[boundaries_fp], boundaries_pred, "orange", "exceed")
    if boundaries_fn is not None:
        _draw_circles(positions[boundaries_fn], boundaries_gt, "grey", "miss")
    plt.xlim(-1, boundaries_gt.shape[0])
    plt.ylim(y_min, y_max)
    plt.grid(axis="y")
    plt.legend()
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return fig


def probs_to_figure(
        probs_gt: np.ndarray, probs_pred: np.ndarray,
        title=None
):
    fig = plt.figure(figsize=(12, 6))
    probs_concat = np.concatenate([np.abs(probs_pred - probs_gt), probs_gt, probs_pred], axis=1)
    plt.pcolormesh(probs_concat.T, vmin=0, vmax=1)
    T, C = probs_gt.shape
    plt.yticks([2.5 * C, 1.5 * C, 0.5 * C], ["pred", "gt", "diff"])
    plt.hlines([C, 2 * C], xmin=0, xmax=T, color="white", linewidth=1.5)
    if title is not None:
        plt.title(title, fontsize=15)
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
