from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from plot_twitter2020_empirical_results import (
    build_distribution_source,
    load_layer_path_source,
)


ROOT = Path(__file__).resolve().parents[2]
ANES_SOURCE = (
    ROOT
    / "outputs"
    / "empirical"
    / "anes2020"
    / "selection_layers"
    / "figures"
    / "source_data"
    / "anes2020_fx_fa_smoothed_distributions.csv"
)
FIGURE_DIR = ROOT / "outputs" / "empirical" / "combined" / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

ANES_ORDER = ["F_X", "F_A_online_political_participants"]
ANES_LABELS = {
    "F_X": r"$F_X$",
    "F_A_online_political_participants": r"$F_A$",
}
ANES_COLORS = {
    "F_X": "#8AB8E2",
    "F_A_online_political_participants": "#60B071",
}
TWITTER_ORDER = ["F_Y", "F_P", "F_V_RT"]
TWITTER_LABELS = {
    "F_Y": r"$F_Y$",
    "F_P": r"$F_P$",
    "F_V_RT": r"$F_V^{\mathrm{RT}}$",
}
TWITTER_COLORS = {
    "F_Y": "#F3A36D",
    "F_P": "#CA6464",
    "F_V_RT": "#81BDB9",
}
SAMPLE_COLORS = {
    "All usable tweets": "#84B4E0",
    r"Confidence $\geq 0.75$": "#E16B6B",
}
SAMPLE_LABELS = {
    "All usable tweets": "Main sample",
    r"Confidence $\geq 0.75$": "High-confidence tweets",
}
GRID_COLOR = "#E3E8EC"
MIDPOINT_COLOR = "#9AA4AE"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 330,
            "savefig.dpi": 600,
        }
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not ANES_SOURCE.exists():
        raise FileNotFoundError(
            "Run plot_anes2020_selection_results.py before creating the combined figure."
        )

    anes = pd.read_csv(ANES_SOURCE)
    # Rebuild the Twitter distribution source so F_Y always uses the current
    # author-uniform tweet weights rather than a potentially stale CSV cache.
    twitter = build_distribution_source()
    twitter_paths = load_layer_path_source()
    return anes, twitter, twitter_paths


def draw_distribution(
    ax: plt.Axes,
    source: pd.DataFrame,
    layer_column: str,
    order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    for layer in order:
        layer_source = source.loc[source[layer_column].eq(layer)]
        color = colors[layer]
        ax.fill_between(
            layer_source["opinion"],
            layer_source["density"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            layer_source["opinion"],
            layer_source["density"],
            color=color,
            lw=2.2,
            label=labels[layer],
        )

    ax.axvline(0.5, color=MIDPOINT_COLOR, lw=0.9, ls="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Density")
    ax.grid(axis="y", color=GRID_COLOR, ls="--", lw=0.6, alpha=0.75)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.97, 1),
        ncol=1,
        frameon=False,
        handlelength=1.9,
        handletextpad=0.5,
        borderaxespad=0.0,
    )


def draw_layer_path(ax: plt.Axes, source: pd.DataFrame, metric: str) -> None:
    layer_order = ["F_Y_author_uniform", "F_P_post_uniform", "F_V_retweet_proxy"]
    styles = {
        "All usable tweets": {"offset": -0.035, "marker": "o", "linestyle": "-"},
        r"Confidence $\geq 0.75$": {
            "offset": 0.035,
            "marker": "s",
            "linestyle": "--",
        },
    }
    x = np.arange(len(layer_order), dtype=float)

    for analysis, style in styles.items():
        selected = (
            source.loc[
                source["metric"].eq(metric) & source["analysis"].eq(analysis)
            ]
            .set_index("layer")
            .reindex(layer_order)
        )
        estimate = selected["estimate"].to_numpy(float)
        lower = selected["ci95_low"].to_numpy(float)
        upper = selected["ci95_high"].to_numpy(float)
        x_values = x + style["offset"]
        color = SAMPLE_COLORS[analysis]
        ax.plot(
            x_values,
            estimate,
            color=color,
            lw=1.8,
            ls=style["linestyle"],
            zorder=2,
        )
        ax.errorbar(
            x_values,
            estimate,
            yerr=np.vstack([estimate - lower, upper - estimate]),
            fmt=style["marker"],
            color=color,
            ecolor=color,
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.35,
            capsize=3.0,
            capthick=1.1,
            label=SAMPLE_LABELS[analysis],
            zorder=3,
        )

    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel(r"$P$" if metric == "P" else r"$\mathrm{BPW}$")
    if metric == "P":
        ax.set_ylim(0.32, 0.40)
        ax.set_yticks([0.32, 0.34, 0.36, 0.38, 0.40])
    else:
        ax.set_ylim(0.48, 0.66)
        ax.set_yticks([0.48, 0.52, 0.56, 0.60, 0.64])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color=GRID_COLOR, ls="--", lw=0.6, alpha=0.8)
    if metric == "P":
        ax.legend(
            loc="upper left",
            ncol=1,
            frameon=False,
            handletextpad=0.1,
            borderaxespad=0.1,
        )


def save_figure(fig: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = "anes_twitter_empirical_results_composite"
    for suffix in ["png", "jpg", "pdf", "svg"]:
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def main() -> None:
    configure_style()
    anes, twitter, twitter_paths = load_sources()

    # The panels are complementary empirical chain segments, not one matched sample.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 7.0),
        sharex="col",
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    draw_distribution(
        axes[0, 0],
        anes,
        "layer",
        ANES_ORDER,
        ANES_LABELS,
        ANES_COLORS,
    )
    draw_distribution(
        axes[1, 0],
        twitter,
        "layer_key",
        TWITTER_ORDER,
        TWITTER_LABELS,
        TWITTER_COLORS,
    )
    draw_layer_path(axes[0, 1], twitter_paths, "P")
    draw_layer_path(axes[1, 1], twitter_paths, "BPW")

    axes[0, 0].tick_params(axis="x", which="both", labelbottom=False)
    axes[1, 0].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axes[1, 0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[1, 0].set_xlabel("Opinion")
    axes[0, 1].tick_params(axis="x", which="both", labelbottom=False)
    axes[1, 1].set_xticks([0.0, 1.0, 2.0])
    axes[1, 1].set_xticklabels(
        [r"$F_Y$", r"$F_P$", r"$F_V^{\mathrm{RT}}$"]
    )
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.98,
        hspace=0.10,
        wspace=0.24,
    )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    combined_source = pd.concat(
        [
            anes.assign(dataset="ANES 2020").rename(columns={"layer": "layer_key"}),
            twitter.assign(dataset="Twitter 2020"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_source.to_csv(
        SOURCE_DIR / "anes_twitter_empirical_layer_distributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    twitter_paths.to_csv(
        SOURCE_DIR / "twitter2020_empirical_layer_paths.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_figure(fig)
    plt.close(fig)
    print(f"Saved combined empirical figure to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
