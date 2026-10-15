from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "outputs" / "full_ablation_new_baseline"
FIGURE_DIR = ROOT / "outputs" / "results_candidate_figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

VARIANT_ORDER = [
    "random_selection",
    "honest_expression",
    "matched_equal_rate",
    "equal_posts",
    "random_visibility",
]

VARIANT_LABELS = {
    "random_selection": "RS",
    "honest_expression": "HE",
    "matched_equal_rate": "MER",
    "equal_posts": "EP",
    "random_visibility": "RV",
}

METRIC_SPECS = [
    ("B_selection_P", r"$\Delta B_{\mathrm{selection}}^{P}$", "#4E79A7"),
    ("B_expression_P", r"$\Delta B_{\mathrm{expression}}^{P}$", "#E15759"),
    ("B_activity_P", r"$\Delta B_{\mathrm{activity}}^{P}$", "#59A14F"),
    ("B_algorithm_P", r"$\Delta B_{\mathrm{algorithm}}^{P}$", "#B07AA1"),
]

ZERO_COLOR = "#1F2933"
GRID_COLOR = "#E6E9ED"


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
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 8.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def ensure_dirs():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem):
    ensure_dirs()
    for suffix in ["jpg", "png", "pdf", "svg"]:
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def load_delta_stats():
    mean_path = ABLATION_DIR / "paired_effects_vs_full_mean.csv"
    ci_path = ABLATION_DIR / "paired_effects_vs_full_ci95.csv"
    if not mean_path.exists() or not ci_path.exists():
        raise FileNotFoundError("Paired ablation effect files are missing.")

    mean_df = pd.read_csv(mean_path)
    ci_df = pd.read_csv(ci_path)
    metric_col = ci_df.columns[1]

    rows = []
    for variant in VARIANT_ORDER:
        mean_row = mean_df[mean_df["variant"] == variant]
        if mean_row.empty:
            continue
        mean_row = mean_row.iloc[0]
        for metric, metric_label, color in METRIC_SPECS:
            ci_row = ci_df[(ci_df["variant"] == variant) & (ci_df[metric_col] == metric)]
            if ci_row.empty:
                continue
            ci_row = ci_row.iloc[0]
            mean_diff = float(mean_row[metric])
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "metric": metric,
                    "metric_label": metric_label,
                    "color": color,
                    "mean_difference": mean_diff,
                    "ci95_low": float(ci_row["ci95_low"]),
                    "ci95_high": float(ci_row["ci95_high"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DIR / "figure_r4b_ablation_delta_mechanisms_single_panel_source_data.csv", index=False)
    return out


def compute_xlim(stats):
    max_abs = max(
        float(np.abs(stats["ci95_low"]).max()),
        float(np.abs(stats["ci95_high"]).max()),
    )
    return 1.08 * max_abs


def plot_grouped_point_range(ax, stats):
    base_y = np.arange(len(VARIANT_ORDER))[::-1]
    y_map = {variant: y for variant, y in zip(VARIANT_ORDER, base_y)}
    offsets = np.array([0.24, 0.08, -0.08, -0.24])

    for offset, (metric, metric_label, color) in zip(offsets, METRIC_SPECS):
        sub = stats[stats["metric"] == metric].copy()
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("variant").reset_index(drop=True)

        for _, row in sub.iterrows():
            y = y_map[row["variant"]] + offset
            mean = float(row["mean_difference"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=6.6,
                mfc=color,
                mec="white",
                mew=0.5,
                color=color,
                ecolor=color,
                elinewidth=0.95,
                capsize=2.4,
                zorder=3,
            )

    ax.axvline(0.0, color=ZERO_COLOR, lw=0.9, ls=(0, (1.5, 2.5)), zorder=0)
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7, zorder=0)
    ax.set_yticks(base_y, [VARIANT_LABELS.get(v, v) for v in VARIANT_ORDER])
    ax.set_xlabel("Change relative to Full-M0")
    ax.set_ylabel("Ablation variant")

    handles = []
    labels = []
    for _metric, metric_label, color in METRIC_SPECS:
        handle = plt.Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=1.0,
        )
        handles.append(handle)
        labels.append(metric_label)
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower right",
        columnspacing=1.0,
        handletextpad=0.5,
    )


def main():
    configure_style()
    ensure_dirs()
    stats = load_delta_stats()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    plot_grouped_point_range(ax, stats)
    xlim = compute_xlim(stats)
    ax.set_xlim(-xlim, xlim)
    save_figure(fig, "figure_r4b_ablation_delta_mechanisms_single_panel")
    plt.close(fig)
    print(f"Saved single-panel Figure 4B to: {FIGURE_DIR}")
    print(
        "Source data saved to: "
        f"{SOURCE_DIR / 'figure_r4b_ablation_delta_mechanisms_single_panel_source_data.csv'}"
    )


if __name__ == "__main__":
    main()
