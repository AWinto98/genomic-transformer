import gzip
import json
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "data" / "processed" / "ctcf_gm12878"
PEAKS_PATH = ROOT / "data" / "raw" / "ctcf_gm12878" / "ENCFF017XLW.bed.gz"
DPI = 200

INK = "#172B3A"
MUTED = "#5B6770"
GRID = "#D9DEE3"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
LIGHT_BLUE = "#56B4E9"
GRAY = "#6E7480"
NUCLEOTIDE_COLORS = {"A": GREEN, "C": BLUE, "G": GOLD, "T": ORANGE}
BASES = "ACGT"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "xtick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.color": MUTED,
            "ytick.labelsize": 9,
            "ytick.major.width": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def style_axes(ax, horizontal_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, pad=4)
    if horizontal_grid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    else:
        ax.grid(False)


def add_axes_header(ax, title, subtitle):
    ax.set_title(
        title,
        loc="left",
        fontsize=15.5,
        fontweight="semibold",
        color=INK,
        pad=35,
    )
    ax.text(
        0,
        1.018,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )


def add_figure_header(fig, title, subtitle, left):
    fig.suptitle(
        title,
        x=left,
        y=0.992,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="semibold",
        color=INK,
    )
    fig.text(left, 0.928, subtitle, ha="left", va="top", fontsize=10, color=MUTED)


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_learning_curves():
    runs = [
        ("transformer_cloud_training_log.jsonl", "d128 · 20 ep", LIGHT_BLUE),
        ("transformer_cloud-002_training_log.jsonl", "d128 · 40 ep", GRAY),
        ("transformer_cloud-004_training_log.jsonl", "d256 · seed 42", BLUE),
        ("transformer_cloud-004-s43_training_log.jsonl", "d256 · seed 43", ORANGE),
        ("transformer_cloud-004-s44_training_log.jsonl", "d256 · seed 44", GREEN),
    ]
    with (DATA_DIR / "baseline_metrics.json").open() as handle:
        baseline = json.load(handle)["metrics"]["validation"]["auroc"]

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=DPI, constrained_layout=True)
    max_epoch = 1
    for filename, label, color in runs:
        rows = read_jsonl(DATA_DIR / filename)
        epochs = np.array([row["epoch"] for row in rows])
        values = np.array([row["val_auroc"] for row in rows])
        best_index = int(np.argmax(values))
        best_epoch = int(epochs[best_index])
        best_value = float(values[best_index])
        max_epoch = max(max_epoch, int(epochs.max()))
        ax.plot(
            epochs,
            values,
            color=color,
            linewidth=1.65,
            label=f"{label} · best {best_value:.3f} · epoch {best_epoch}",
            zorder=3,
        )
        ax.plot(
            [best_epoch],
            [best_value],
            linestyle="none",
            marker="o",
            markersize=6.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=4,
        )

    ax.axhline(
        baseline,
        color=MUTED,
        linewidth=1.25,
        linestyle=(0, (1.5, 2.5)),
        label=f"6-mer logistic baseline · {baseline:.3f}",
        zorder=2,
    )
    ax.set_xlim(0.5, max_epoch + 1.5)
    ax.set_ylim(0.50, 0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.legend(
        loc="lower right",
        frameon=False,
        handlelength=2.3,
        labelspacing=0.45,
        borderaxespad=0.5,
    )
    style_axes(ax)
    add_axes_header(
        ax,
        "Larger transformers improve validation AUROC",
        "Held-out chr17 · scratch models · markers show each run's best epoch",
    )
    save_figure(fig, FIGURE_DIR / "learning_curves.png")


@lru_cache(maxsize=4)
def nucleotide_path(base):
    return TextPath(
        (0, 0),
        base,
        size=1,
        prop=FontProperties(family="DejaVu Sans", weight="bold"),
    )


def load_attribution(npy_path, json_path):
    matrix = np.load(npy_path, allow_pickle=False)
    with json_path.open() as handle:
        metadata = json.load(handle)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"Expected an N x 4 attribution matrix: {npy_path}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"Attribution values must be finite and nonnegative: {npy_path}")
    offsets = np.arange(-40, 41)
    center = matrix.shape[0] // 2
    cropped = matrix[center + offsets]
    denominator = float(cropped.sum(axis=1).max())
    if denominator <= 0:
        raise ValueError(f"Cropped attribution has zero total mass: {npy_path}")
    return cropped, denominator, metadata


def add_glyph(ax, base, position, bottom, height):
    if height <= 0:
        return
    path = nucleotide_path(base)
    bounds = path.get_extents()
    transform = (
        Affine2D()
        .translate(-bounds.x0, -bounds.y0)
        .scale(0.86 / bounds.width, height / bounds.height)
        .translate(position - 0.43, bottom)
    )
    patch = PathPatch(
        path,
        transform=transform + ax.transData,
        facecolor=NUCLEOTIDE_COLORS[base],
        edgecolor="none",
        linewidth=0,
        zorder=3,
    )
    ax.add_patch(patch)


def draw_attribution_logo(ax, cropped, denominator, y_label=None):
    offsets = np.arange(-40, 41)
    for position, column in zip(offsets, cropped):
        bottom = 0.0
        for base_index in np.argsort(column):
            height = float(column[base_index]) / denominator
            add_glyph(ax, BASES[base_index], position, bottom, height)
            bottom += height
    ax.set_xlim(-41.5, 41.5)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if y_label:
        ax.set_ylabel(y_label)
    style_axes(ax)


def render_ctcf_motif():
    cropped, denominator, metadata = load_attribution(
        FIGURE_DIR / "ctcf_motif.npy", FIGURE_DIR / "ctcf_motif.json"
    )
    fig, ax = plt.subplots(figsize=(9, 4), dpi=DPI, constrained_layout=True)
    draw_attribution_logo(
        ax,
        cropped,
        denominator,
        y_label="Relative attribution\n(maximum position = 1)",
    )
    ax.set_xlabel("Position relative to window center (bp)")
    max_offset = int(metadata["peak_offset_of_max_importance"])
    ax.axvline(
        max_offset,
        color=MUTED,
        linewidth=0.8,
        linestyle=(0, (2, 3)),
        alpha=0.65,
        zorder=1,
    )
    ax.annotate(
        f"maximum importance offset: {max_offset:+d} bp",
        xy=(max_offset, 1.0),
        xytext=(7, -3),
        textcoords="offset points",
        fontsize=8.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    add_axes_header(
        ax,
        "CTCF attribution recovered by gradient × input saliency",
        f"Scratch d256 transformer · {int(metadata['n_windows']):,} held-out chr17 positive windows",
    )
    save_figure(fig, FIGURE_DIR / "ctcf_motif.png")


def render_interpret_comparison():
    panel_specs = [
        (
            "ctcf_motif.npy",
            "ctcf_motif.json",
            "Gradient × input · scratch",
        ),
        (
            "scratch_rollout_rollout.npy",
            "scratch_rollout_rollout.json",
            "Attention rollout · scratch",
        ),
        (
            "mlm_rollout_rollout.npy",
            "mlm_rollout_rollout.json",
            "Attention rollout · pilot MLM",
        ),
    ]
    panels = []
    for npy_name, json_name, title in panel_specs:
        cropped, denominator, metadata = load_attribution(
            FIGURE_DIR / npy_name, FIGURE_DIR / json_name
        )
        panels.append((cropped, denominator, metadata, title))
    n_windows = int(panels[0][2]["n_windows"])
    if any(int(panel[2]["n_windows"]) != n_windows for panel in panels[1:]):
        raise ValueError("All comparison panels must describe the same held-out windows")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 5),
        dpi=DPI,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.74, wspace=0.12)
    for index, (ax, (cropped, denominator, _, title)) in enumerate(zip(axes, panels)):
        draw_attribution_logo(
            ax,
            cropped,
            denominator,
            y_label="Relative attribution\n(panel maximum = 1)" if index == 0 else None,
        )
        ax.set_title(title, fontsize=11, color=INK, pad=10)
    fig.supxlabel("Position relative to window center (bp)", fontsize=10.5, color=INK, y=0.055)
    add_figure_header(
        fig,
        "CTCF motif recovery across attribution methods",
        f"{n_windows:,} held-out chr17 positives per panel · central ±40 bp · panels normalized independently",
        left=0.065,
    )
    save_figure(fig, FIGURE_DIR / "interpret_comparison.png")


def centered_rolling_median(values, window=21):
    radius = window // 2
    padded = np.pad(values, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def load_inference_data():
    rows = []
    with (DATA_DIR / "demo_chr8_scan.bed").open() as handle:
        for line in handle:
            if line.strip():
                fields = line.rstrip("\n").split("\t")
                rows.append((fields[0], int(fields[1]), int(fields[2]), float(fields[4])))
    if len(rows) != 2000:
        raise ValueError(f"Expected exactly 2,000 scan tiles, found {len(rows):,}")
    chromosomes = {row[0] for row in rows}
    if chromosomes != {"chr8"}:
        raise ValueError("The inference scan must contain only chr8 tiles")
    starts = np.array([row[1] for row in rows])
    ends = np.array([row[2] for row in rows])
    scores = np.array([row[3] for row in rows])
    if not ((ends - starts == 1000).all() and (np.diff(starts) == 1000).all()):
        raise ValueError("Expected non-overlapping, consecutive 1 kb scan tiles")

    scan_start = int(starts[0])
    scan_end = int(ends[-1])
    peaks = []
    with gzip.open(PEAKS_PATH, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "chr8":
                continue
            peak_start, peak_end = int(fields[1]), int(fields[2])
            if peak_start < scan_end and peak_end > scan_start:
                peaks.append((peak_start, peak_end))
    peak_array = np.asarray(peaks, dtype=int).reshape(-1, 2)
    overlaps = (
        (starts[:, None] < peak_array[None, :, 1])
        & (peak_array[None, :, 0] < ends[:, None])
    ).any(axis=1)
    threshold = float(np.quantile(scores, 0.9))
    top_decile = scores >= threshold
    return starts, ends, scores, peak_array, overlaps, threshold, top_decile


def render_inference_demo():
    starts, ends, scores, peaks, overlaps, threshold, top_decile = load_inference_data()
    positions = (starts + ends) / 2 / 1_000_000
    rolling_median = centered_rolling_median(scores, window=21)
    top_rate = float(overlaps[top_decile].mean())
    baseline_rate = float(overlaps.mean())

    fig = plt.figure(figsize=(9, 4.5), dpi=DPI, constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[5.2, 0.75, 0.75],
        hspace=0.16,
        left=0.12,
        right=0.985,
        bottom=0.13,
        top=0.74,
    )
    ax = fig.add_subplot(grid[0])
    tile_ax = fig.add_subplot(grid[1], sharex=ax)
    peak_ax = fig.add_subplot(grid[2], sharex=ax)

    raw_line = ax.plot(
        positions,
        scores,
        color=LIGHT_BLUE,
        linewidth=0.65,
        alpha=0.8,
        label="Raw probability",
        zorder=2,
    )[0]
    median_line = ax.plot(
        positions,
        rolling_median,
        color=BLUE,
        linewidth=1.8,
        label="Centered rolling median (21 windows · 21 kb)",
        zorder=3,
    )[0]
    threshold_line = ax.axhline(
        threshold,
        color=GOLD,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        label=f"Top-decile threshold ({threshold:.3f})",
        zorder=1,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Prediction probability")
    ax.tick_params(labelbottom=False)
    style_axes(ax)
    ax.legend(
        handles=[raw_line, median_line, threshold_line],
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
        borderaxespad=0,
    )
    fig.text(
        0.985,
        0.865,
        f"Top-decile peak overlap: {top_rate:.1%} vs {baseline_rate:.1%} baseline\n"
        f"{len(peaks)} ENCODE CTCF peaks in scan",
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
        linespacing=1.35,
    )

    top_positions = positions[top_decile]
    top_colors = np.where(overlaps[top_decile], ORANGE, GRAY)
    tile_ax.scatter(
        top_positions,
        np.full(top_positions.shape, 0.5),
        marker="|",
        s=85,
        linewidths=1.15,
        c=top_colors,
    )
    tile_ax.set_ylim(0, 1)
    tile_ax.set_yticks([])
    tile_ax.set_ylabel("Top-decile\ntiles", rotation=0, ha="right", va="center", fontsize=9)
    tile_ax.tick_params(labelbottom=False, length=3)
    tile_ax.text(
        0.995,
        0.95,
        "orange: tile overlaps a peak",
        transform=tile_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    style_axes(tile_ax, horizontal_grid=False)
    tile_ax.spines["left"].set_visible(False)
    tile_ax.spines["bottom"].set_visible(False)

    peak_midpoints = peaks.mean(axis=1) / 1_000_000
    peak_ax.scatter(
        peak_midpoints,
        np.full(peak_midpoints.shape, 0.5),
        marker="|",
        s=100,
        linewidths=1.3,
        color=ORANGE,
    )
    peak_ax.set_ylim(0, 1)
    peak_ax.set_yticks([])
    peak_ax.set_ylabel(
        "ENCODE CTCF\npeak midpoints", rotation=0, ha="right", va="center", fontsize=9
    )
    peak_ax.set_xlabel("Genomic position on chr8 (Mb)")
    peak_ax.text(
        0.995,
        0.95,
        "markers not to scale",
        transform=peak_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    style_axes(peak_ax, horizontal_grid=False)
    peak_ax.spines["left"].set_visible(False)
    peak_ax.set_xlim(starts[0] / 1_000_000, ends[-1] / 1_000_000)
    peak_ax.set_xticks([22, 22.5, 23, 23.5, 24])

    add_figure_header(
        fig,
        "Held-out chr8 scan: high scores rarely coincide with CTCF peaks",
        f"chr8:22–24 Mb · {len(scores):,} non-overlapping 1 kb tiles · scratch d256 seed 42",
        left=0.12,
    )
    save_figure(fig, FIGURE_DIR / "inference_demo.png")


def main():
    configure_style()
    render_learning_curves()
    render_ctcf_motif()
    render_interpret_comparison()
    render_inference_demo()


if __name__ == "__main__":
    main()
