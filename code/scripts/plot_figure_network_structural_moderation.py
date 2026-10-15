from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NETWORK_PATH = (
    ROOT / "outputs" / "network_comparison" / "summary_by_replication.csv"
)
FIGURE_DIR = ROOT / "outputs" / "results_candidate_figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

EXPECTED_REPLICATIONS = 30
Z_95 = 1.96

NETWORK_ORDER = ["BA", "ER", "WS", "BA_HOMOPHILY"]
NETWORK_LABELS = {
    "BA": "BA",
    "ER": "ER",
    "WS": "WS",
    "BA_HOMOPHILY": "BA-Homophily",
}

METRIC_SPECS = [
    ("VRB_P", r"$\mathrm{VRB}_{P}$"),
    ("B_reach_P", r"$B_{\mathrm{rch}}^{P}$"),
    ("B_algorithm_P", r"$B_{\mathrm{alg}}^{P}$"),
]
METRIC_COLORS = {
    "VRB_P": "#F8CBAD",
    "B_reach_P": "#D37F7F",
    "B_algorithm_P": "#9DC3E6",
}

ZERO_COLOR = "#8E99A3"
GRID_COLOR = "#E8ECF0"
DARK = "#1F2933"


def configure_style():
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
            "axes.labelsize": 12.5,
            "xtick.direction": "in",
            "ytick.direction": "out",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 10,
            "figure.dpi": 330,
            "savefig.dpi": 600,
        }
    )


def ensure_dirs():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def load_network_results():
    if not NETWORK_PATH.exists():
        raise FileNotFoundError(f"Required network result file is missing: {NETWORK_PATH}")

    required = {
        "network",
        "replication_id",
        "VRB_P",
        "B_reach_P",
        "B_algorithm_P",
    }
    frame = pd.read_csv(NETWORK_PATH)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Network result file is missing columns: {missing}")

    frame = frame.copy()
    frame["replication_id"] = pd.to_numeric(
        frame["replication_id"], errors="raise"
    ).astype(int)
    for metric, _ in METRIC_SPECS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")

    rows = []
    for network in NETWORK_ORDER:
        selected = frame.loc[frame["network"] == network].copy()
        if selected.empty:
            raise ValueError(f"Network '{network}' was not found in {NETWORK_PATH.name}.")
        if selected["replication_id"].duplicated().any():
            raise ValueError(f"Network '{network}' contains duplicate replication IDs.")
        if selected["replication_id"].nunique() != EXPECTED_REPLICATIONS:
            raise ValueError(
                f"Network '{network}' contains "
                f"{selected['replication_id'].nunique()} replications; "
                f"expected {EXPECTED_REPLICATIONS}."
            )

        for metric, metric_label in METRIC_SPECS:
            values = selected[metric]
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    "network": network,
                    "network_label": NETWORK_LABELS[network],
                    "metric": metric,
                    "metric_label": metric_label,
                    "n": int(len(values)),
                    "mean": mean,
                    "sem": sem,
                    "ci95_low": mean - Z_95 * sem,
                    "ci95_high": mean + Z_95 * sem,
                }
            )

    return pd.DataFrame(rows)


def style_axis(ax):
    ax.axhline(0.0, color=ZERO_COLOR, lw=0.8, zorder=3)
    #ax.grid(axis="y", color=GRID_COLOR, lw=0.8, ls="--", zorder=0)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    for spine in ax.spines.values():
        spine.set_color(DARK)


def draw_grouped_bars(ax, source_data):
    x_positions = np.arange(len(NETWORK_ORDER), dtype=float)
    bar_width = 0.22
    offsets = [-bar_width, 0.0, bar_width]

    for offset, (metric, metric_label) in zip(offsets, METRIC_SPECS):
        subset = source_data.loc[source_data["metric"] == metric].copy()
        subset["network"] = pd.Categorical(
            subset["network"], categories=NETWORK_ORDER, ordered=True
        )
        subset = subset.sort_values("network").reset_index(drop=True)

        means = subset["mean"].to_numpy(dtype=float)
        lower_errors = means - subset["ci95_low"].to_numpy(dtype=float)
        upper_errors = subset["ci95_high"].to_numpy(dtype=float) - means
        color = METRIC_COLORS[metric]

        ax.bar(
            x_positions + offset,
            means,
            width=bar_width * 0.88,
            color=color,
            edgecolor="none",
            linewidth=0.7,
            label=metric_label,
            zorder=3,
        )
        ax.errorbar(
            x_positions + offset,
            means,
            yerr=[lower_errors, upper_errors],
            fmt="none",
            ecolor=color,
            elinewidth=1,
            capsize=3.0,
            capthick=1.2,
            zorder=4,
        )

    low = min(0.0, float(source_data["ci95_low"].min()))
    high = max(0.0, float(source_data["ci95_high"].max()))
    span = high - low
    ax.set_ylim(low - 0.10 * span, high + 0.13 * span)
    ax.set_xlim(-0.55, len(NETWORK_ORDER) - 0.45)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([NETWORK_LABELS[item] for item in NETWORK_ORDER])
    ax.set_ylabel("Polarization bias", labelpad=3,fontsize=12.5)
    style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.35, 0.02),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.3,
        columnspacing=1,
    )


def save_outputs(fig, source_data):
    ensure_dirs()
    stem = "figure_network_structural_moderation"
    source_data.to_csv(
        SOURCE_DIR / f"{stem}_source_data.csv",
        index=False,
    )

    for suffix in ["jpg", "png", "pdf", "svg"]:
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
        )
    fig.savefig(
        FIGURE_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main():
    configure_style()
    source_data = load_network_results()

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    draw_grouped_bars(ax, source_data)

    save_outputs(fig, source_data)
    plt.close(fig)

    print(f"Saved Figure Y to: {FIGURE_DIR}")
    print(
        "Source data: "
        f"{SOURCE_DIR / 'figure_network_structural_moderation_source_data.csv'}"
    )


if __name__ == "__main__":
    main()
