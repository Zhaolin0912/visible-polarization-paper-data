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
    "full",
    "random_selection",
    "honest_expression",
    "matched_equal_rate",
    "equal_posts",
    "random_visibility",
]

VARIANT_LABELS = {
    "full": "LVBM",
    "random_selection": "RS",
    "honest_expression": "HE",
    "matched_equal_rate": "MER",
    "equal_posts": "EP",
    "random_visibility": "RV",
}

METRIC_SPECS = [
    ("EPRB_actual", r"$EPRB$", "#74BA83"),
    ("VRB_P_actual", r"$\mathrm{VRB}_P$", "#84B4E0"),
]


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11,
            "figure.dpi": 330,
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


def load_ablation_stats():
    mean_path = ABLATION_DIR / "report_mean.csv"
    sem_path = ABLATION_DIR / "report_sem.csv"
    if not mean_path.exists() or not sem_path.exists():
        raise FileNotFoundError("Ablation report_mean.csv or report_sem.csv is missing.")

    mean_df = pd.read_csv(mean_path)
    sem_df = pd.read_csv(sem_path)
    frame = mean_df.merge(sem_df, on="variant", suffixes=("_mean", "_sem"))
    frame = frame[frame["variant"].isin(VARIANT_ORDER)].copy()
    frame["variant"] = pd.Categorical(frame["variant"], categories=VARIANT_ORDER, ordered=True)
    frame = frame.sort_values("variant").reset_index(drop=True)
    return frame


def build_long_stats(frame):
    rows = []
    for _, row in frame.iterrows():
        variant = row["variant"]
        for metric, label, color in METRIC_SPECS:
            mean = float(row[f"{metric}_mean"])
            sem = float(row[f"{metric}_sem"])
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "metric": metric,
                    "metric_label": label,
                    "color": color,
                    "mean": mean,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DIR / "figure_r4a_ablation_dumbbell_source_data.csv", index=False)
    return out


def plot_dumbbell(ax, stats):
    y_positions = np.arange(len(VARIANT_ORDER))[::-1]
    y_map = {variant: y for variant, y in zip(VARIANT_ORDER, y_positions)}

    for variant in VARIANT_ORDER:
        sub = stats[stats["variant"] == variant].copy()
        if sub.empty:
            continue
        sub = sub.set_index("metric").loc[[m for m, _label, _color in METRIC_SPECS]].reset_index()
        y = y_map[variant]
        x0 = float(sub.loc[sub["metric"] == METRIC_SPECS[0][0], "mean"].iloc[0])
        x1 = float(sub.loc[sub["metric"] == METRIC_SPECS[1][0], "mean"].iloc[0])

        ax.plot([x0, x1], [y, y], color="#C9D2DB", lw=1.35, zorder=1)

        for _, row in sub.iterrows():
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            color = row["color"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=7.2,
                mfc=color,
                mec="white",
                mew=0.6,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.8,
                zorder=3,
            )

    ax.set_yticks(y_positions, [VARIANT_LABELS.get(v, v) for v in VARIANT_ORDER])
    ax.set_xlabel("Bias", fontsize=14.5, fontfamily="Times New Roman")
    ax.axvline(0.0, color="#E6E9ED", lw=1.2, zorder=0)
    ax.grid(axis="x", color="#E6E9ED", lw=0.9, zorder=0,ls="--")

    handles = []
    labels = []
    for _metric, label, color in METRIC_SPECS:
        handle = plt.Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            lw=1.0,
        )
        handles.append(handle)
        labels.append(label)
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower right",
        prop={"family": "Times New Roman", "size": mpl.rcParams["legend.fontsize"]},
    )

    # Keep all axis and tick text in Times New Roman in every export format.
    ax.xaxis.label.set_fontfamily("Times New Roman")
    ax.yaxis.label.set_fontfamily("Times New Roman")
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontfamily("Times New Roman")


def main():
    configure_style()
    ensure_dirs()
    frame = load_ablation_stats()
    stats = build_long_stats(frame)

    fig, ax = plt.subplots(figsize=(4, 4.5), constrained_layout=True)
    plot_dumbbell(ax, stats)
    save_figure(fig, "figure_r4a_ablation_dumbbell")
    #plt.close(fig)
    print(f"Saved Figure 4A ablation dumbbell to: {FIGURE_DIR}")
    print(f"Source data saved to: {SOURCE_DIR / 'figure_r4a_ablation_dumbbell_source_data.csv'}")


if __name__ == "__main__":
    main()
