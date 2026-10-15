from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m0_lps_epo.config import load_config


BASELINE_DIR = ROOT / "outputs" / "baseline"
FIGURE_DIR = BASELINE_DIR / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"


PALETTE = {
    "blue": "#3B6EA8",
    "teal": "#2A9D8F",
    "orange": "#D98C32",
    "red": "#C85C5C",
    "purple": "#7A6BB7",
    "gray": "#6F7782",
    "light_gray": "#D5DAE0",
    "dark": "#1F2933",
}


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def mean_sem_ci(frame, columns, group_col=None):
    if group_col is None:
        rows = []
        for col in columns:
            values = frame[col].dropna()
            n = len(values)
            mean = float(values.mean())
            sem = float(values.sem()) if n > 1 else 0.0
            rows.append(
                {
                    "metric": col,
                    "n": n,
                    "mean": mean,
                    "sem": sem,
                    "ci95_low": mean - 1.96 * sem,
                    "ci95_high": mean + 1.96 * sem,
                }
            )
        return pd.DataFrame(rows)

    grouped = frame.groupby(group_col, as_index=False)
    parts = []
    for col in columns:
        stats = grouped[col].agg(["mean", "sem", "count"]).reset_index()
        stats["metric"] = col
        stats["ci95_low"] = stats["mean"] - 1.96 * stats["sem"].fillna(0.0)
        stats["ci95_high"] = stats["mean"] + 1.96 * stats["sem"].fillna(0.0)
        stats = stats.rename(columns={"count": "n"})
        parts.append(stats[[group_col, "metric", "n", "mean", "sem", "ci95_low", "ci95_high"]])
    return pd.concat(parts, ignore_index=True)


def add_panel_label(ax, label):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["dark"],
    )


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def plot_time_series(ax, time_stats, metric, color, label):
    sub = time_stats[time_stats["metric"] == metric].sort_values("t")
    x = sub["t"].to_numpy(dtype=float)
    mean = sub["mean"].to_numpy(dtype=float)
    low = sub["ci95_low"].to_numpy(dtype=float)
    high = sub["ci95_high"].to_numpy(dtype=float)
    ax.plot(x, mean, color=color, lw=1.8, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.14, lw=0)


def plot_lollipop_ci(ax, stats, order, labels, colors, x_label, title):
    data = stats.set_index("metric").loc[order].reset_index()
    y = np.arange(len(data))
    means = data["mean"].to_numpy(dtype=float)
    lows = data["ci95_low"].to_numpy(dtype=float)
    highs = data["ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([means - lows, highs - means])

    ax.axvline(0.0, color="#9AA3AD", lw=0.8, zorder=0)
    for yi, value, color in zip(y, means, colors):
        ax.plot([0, value], [yi, yi], color=color, lw=3.0, solid_capstyle="round", alpha=0.9)
    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="o",
        ms=5,
        color=PALETTE["dark"],
        ecolor=PALETTE["dark"],
        elinewidth=0.9,
        capsize=2.5,
        zorder=4,
    )
    for yi, value in zip(y, means):
        ha = "left" if value >= 0 else "right"
        offset = 0.0025 if value >= 0 else -0.0025
        ax.text(value + offset, yi, f"{value:+.3f}", va="center", ha=ha, fontsize=6.5)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color="#E6E9ED", lw=0.7)


def plot_dual_contribution_ci(ax, stats, p_order, bpw_order, labels, colors, summary):
    p_data = stats.set_index("metric").loc[p_order].reset_index()
    bpw_data = stats.set_index("metric").loc[bpw_order].reset_index()
    y = np.arange(len(labels))
    offsets = {"P": -0.14, "BPW": 0.14}

    ax.axvline(0.0, color="#9AA3AD", lw=0.8, zorder=0)
    for idx, color in enumerate(colors):
        p_mean = float(p_data.loc[idx, "mean"])
        bpw_mean = float(bpw_data.loc[idx, "mean"])
        ax.plot([0, p_mean], [y[idx] + offsets["P"], y[idx] + offsets["P"]], color=color, lw=2.7, alpha=0.90)
        ax.plot([0, bpw_mean], [y[idx] + offsets["BPW"], y[idx] + offsets["BPW"]], color=color, lw=2.0, alpha=0.48)

    for data, metric_label, marker, offset, size in [
        (p_data, "P", "o", offsets["P"], 4.8),
        (bpw_data, "BPW", "s", offsets["BPW"], 4.2),
    ]:
        means = data["mean"].to_numpy(dtype=float)
        lows = data["ci95_low"].to_numpy(dtype=float)
        highs = data["ci95_high"].to_numpy(dtype=float)
        ax.errorbar(
            means,
            y + offset,
            xerr=np.vstack([means - lows, highs - means]),
            fmt=marker,
            ms=size,
            color=PALETTE["dark"],
            ecolor=PALETTE["dark"],
            elinewidth=0.85,
            capsize=2.4,
            zorder=4,
            label=metric_label,
        )

    vrb_p = float(summary["VRB_P"].mean())
    vrb_bpw = float(summary["VRB_BPW"].mean())
    ax.axvline(vrb_p, color=PALETTE["blue"], lw=1.0, ls="--", alpha=0.75)
    ax.axvline(vrb_bpw, color=PALETTE["orange"], lw=1.0, ls="--", alpha=0.75)
    ax.text(vrb_p, -0.68, f"net P {vrb_p:+.3f}", ha="center", va="bottom", fontsize=6.5, color=PALETTE["blue"])
    ax.text(vrb_bpw, -0.38, f"net BPW {vrb_bpw:+.3f}", ha="center", va="bottom", fontsize=6.5, color=PALETTE["orange"])

    xmin = min(float(p_data["ci95_low"].min()), float(bpw_data["ci95_low"].min()), 0.0)
    xmax = max(float(p_data["ci95_high"].max()), float(bpw_data["ci95_high"].max()), vrb_bpw, 0.0)
    pad = 0.08 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to visible polarization bias")
    ax.set_title("Sequential accounting decomposition", loc="left", fontweight="bold", pad=3)
    ax.grid(axis="x", color="#E6E9ED", lw=0.7)
    ax.legend(loc="lower right", frameon=False, title="Metric")


def plot_paired_prb(ax, summary):
    pairs = summary[["PRB_R", "PRB_actual"]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(12345)
    x_r = np.zeros(len(pairs)) + rng.normal(0, 0.018, len(pairs))
    x_a = np.ones(len(pairs)) + rng.normal(0, 0.018, len(pairs))
    for idx, row in pairs.iterrows():
        ax.plot([x_r[idx], x_a[idx]], [row["PRB_R"], row["PRB_actual"]], color="#CBD2D9", lw=0.7, zorder=1)
    ax.scatter(x_r, pairs["PRB_R"], s=13, color=PALETTE["gray"], alpha=0.65, edgecolor="none", zorder=2)
    ax.scatter(x_a, pairs["PRB_actual"], s=13, color=PALETTE["teal"], alpha=0.75, edgecolor="none", zorder=2)

    stats = mean_sem_ci(summary, ["PRB_R", "PRB_actual", "EPRB_actual"]).set_index("metric")
    for xpos, metric, color in [(0, "PRB_R", PALETTE["gray"]), (1, "PRB_actual", PALETTE["teal"])]:
        mean = stats.loc[metric, "mean"]
        low = stats.loc[metric, "ci95_low"]
        high = stats.loc[metric, "ci95_high"]
        ax.errorbar(
            xpos,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            ms=5.2,
            color=color,
            ecolor=PALETTE["dark"],
            elinewidth=1.0,
            capsize=3,
            zorder=5,
        )

    eprb_mean = stats.loc["EPRB_actual", "mean"]
    eprb_low = stats.loc["EPRB_actual", "ci95_low"]
    eprb_high = stats.loc["EPRB_actual", "ci95_high"]
    ax.text(
        0.5,
        0.98,
        f"EPRB = {eprb_mean:+.3f}\n95% CI [{eprb_low:+.3f}, {eprb_high:+.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#D9DEE5", linewidth=0.7),
    )
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["Random reach", "Actual feed"])
    ax.set_ylabel("Observer-level PRB")
    ax.set_title("Algorithmic perception bias", loc="left", fontweight="bold", pad=3)
    ax.grid(axis="y", color="#E6E9ED", lw=0.7)


def make_main_figure(summary, time_series):
    time_metrics = ["VRB_P", "VRB_BPW"]
    time_stats = mean_sem_ci(time_series, time_metrics, group_col="t")
    contribution_metrics_p = [
        "B_selection_P",
        "B_expression_P",
        "B_activity_P",
        "B_reach_P",
        "B_algorithm_P",
    ]
    contribution_metrics_bpw = [
        "B_selection_BPW",
        "B_expression_BPW",
        "B_activity_BPW",
        "B_reach_BPW",
        "B_algorithm_BPW",
    ]
    contribution_stats = mean_sem_ci(summary, contribution_metrics_p + contribution_metrics_bpw)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    time_stats.to_csv(SOURCE_DIR / "figure_baseline_main_time_series.csv", index=False)
    contribution_stats.to_csv(SOURCE_DIR / "figure_baseline_main_contributions.csv", index=False)
    mean_sem_ci(summary, ["PRB_R", "PRB_actual", "EPRB_actual"]).to_csv(
        SOURCE_DIR / "figure_baseline_main_prb.csv", index=False
    )

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.08, 1.0], width_ratios=[1.24, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    plot_time_series(ax_a, time_stats, "VRB_P", PALETTE["blue"], "VRB-P")
    plot_time_series(ax_a, time_stats, "VRB_BPW", PALETTE["orange"], "VRB-BPW")
    final_w = int(load_config(ROOT / "configs" / "baseline.yaml").simulation.W)
    max_t = int(time_series["t"].max())
    ax_a.axhline(0, color="#9AA3AD", lw=0.8)
    ax_a.axvspan(max_t - final_w + 1, max_t, color="#E9EEF4", alpha=0.8, lw=0)
    ax_a.text(
        max_t - final_w + 2,
        ax_a.get_ylim()[1] * 0.88,
        "reported\nstable window",
        fontsize=6.5,
        color=PALETTE["gray"],
        va="top",
    )
    ax_a.set_xlabel("Simulation time")
    ax_a.set_ylabel("Visible polarization bias")
    ax_a.set_title("Stable positive visible polarization under the baseline", loc="left", fontweight="bold", pad=4)
    ax_a.legend(loc="upper right", frameon=False, ncol=2)
    ax_a.grid(axis="y", color="#E6E9ED", lw=0.7)
    add_panel_label(ax_a, "A")

    labels = ["Selection", "Expression", "Activity", "Reach", "Algorithm"]
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["teal"], PALETTE["gray"], PALETTE["orange"]]
    plot_dual_contribution_ci(
        ax_b,
        contribution_stats,
        contribution_metrics_p,
        contribution_metrics_bpw,
        labels,
        colors,
        summary,
    )
    add_panel_label(ax_b, "B")

    plot_paired_prb(ax_c, summary)
    add_panel_label(ax_c, "C")

    fig.suptitle(
        "Baseline M0-LPS-EPO visible polarization",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    save_figure(fig, "figure_1_baseline_visible_polarization")
    plt.close(fig)


def make_diagnostic_figure(summary, time_series):
    time_metrics = ["FeedSaturation", "ParticipationRate"]
    time_stats = mean_sem_ci(time_series, time_metrics, group_col="t")
    supply_stats = mean_sem_ci(summary, ["MeanCandidates", "MeanVisibleItems_R", "MeanVisibleItems_V", "TotalPosts"])
    rate_stats = mean_sem_ci(summary, ["ParticipationRate", "FeedSaturation", "ValidObserverRate_V"])
    auxiliary_stats = mean_sem_ci(
        summary,
        ["VRB_U_mu", "VRB_CON", "VRB_BTM", "S_deltaB_actual", "BAL_deltaB_actual", "PRB_sd_actual"],
    )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    time_stats.to_csv(SOURCE_DIR / "figure_s1_baseline_diagnostics_time.csv", index=False)
    supply_stats.to_csv(SOURCE_DIR / "figure_s1_baseline_diagnostics_supply.csv", index=False)
    rate_stats.to_csv(SOURCE_DIR / "figure_s1_baseline_diagnostics_rates.csv", index=False)
    auxiliary_stats.to_csv(SOURCE_DIR / "figure_s1_baseline_diagnostics_auxiliary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_time_series(ax_a, time_stats, "FeedSaturation", PALETTE["teal"], "Feed saturation")
    plot_time_series(ax_a, time_stats, "ParticipationRate", PALETTE["blue"], "Participation rate")
    ax_a.set_xlabel("Simulation time")
    ax_a.set_ylabel("Rate")
    ax_a.set_title("Stable feed pressure and participation", loc="left", fontweight="bold", pad=3)
    ax_a.legend(frameon=False, loc="upper right")
    ax_a.grid(axis="y", color="#E6E9ED", lw=0.7)
    add_panel_label(ax_a, "A")

    supply_order = ["MeanCandidates", "MeanVisibleItems_R", "MeanVisibleItems_V"]
    supply_labels = ["Candidates", "Visible-R", "Visible-V"]
    supply_data = supply_stats.set_index("metric").loc[supply_order].reset_index()
    x = np.arange(len(supply_data))
    means = supply_data["mean"].to_numpy(dtype=float)
    lows = supply_data["ci95_low"].to_numpy(dtype=float)
    highs = supply_data["ci95_high"].to_numpy(dtype=float)
    ax_b.bar(x, means, color=[PALETTE["light_gray"], PALETTE["gray"], PALETTE["orange"]], width=0.62)
    ax_b.errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor=PALETTE["dark"], capsize=3, lw=0.9)
    ax_b.set_xticks(x, supply_labels, rotation=0)
    ax_b.set_ylabel("Items per observer")
    ax_b.set_title("Candidate supply and visible feed size", loc="left", fontweight="bold", pad=3)
    ax_b.grid(axis="y", color="#E6E9ED", lw=0.7)
    add_panel_label(ax_b, "B")

    rate_order = ["ParticipationRate", "FeedSaturation", "ValidObserverRate_V"]
    rate_labels = ["Participation", "Saturation", "Valid observers"]
    rate_data = rate_stats.set_index("metric").loc[rate_order].reset_index()
    x = np.arange(len(rate_data))
    means = rate_data["mean"].to_numpy(dtype=float)
    lows = rate_data["ci95_low"].to_numpy(dtype=float)
    highs = rate_data["ci95_high"].to_numpy(dtype=float)
    ax_c.bar(x, means, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["purple"]], width=0.62)
    ax_c.errorbar(x, means, yerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor=PALETTE["dark"], capsize=3, lw=0.9)
    ax_c.set_ylim(0, min(1.05, max(1.0, highs.max() * 1.15)))
    ax_c.set_xticks(x, rate_labels, rotation=20, ha="right")
    ax_c.set_ylabel("Mean rate")
    ax_c.set_title("Baseline operating regime", loc="left", fontweight="bold", pad=3)
    ax_c.grid(axis="y", color="#E6E9ED", lw=0.7)
    add_panel_label(ax_c, "C")

    aux_order = ["VRB_U_mu", "VRB_CON", "VRB_BTM", "S_deltaB_actual", "BAL_deltaB_actual", "PRB_sd_actual"]
    aux_labels = ["VRB-Umu", "VRB-CON", "VRB-BTM", "Two-end mass", "Balance", "PRB s.d."]
    aux_data = auxiliary_stats.set_index("metric").loc[aux_order].reset_index()
    y = np.arange(len(aux_data))
    means = aux_data["mean"].to_numpy(dtype=float)
    lows = aux_data["ci95_low"].to_numpy(dtype=float)
    highs = aux_data["ci95_high"].to_numpy(dtype=float)
    ax_d.axvline(0, color="#9AA3AD", lw=0.8)
    ax_d.barh(y, means, color=[PALETTE["blue"], PALETTE["gray"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"], PALETTE["red"]], height=0.58)
    ax_d.errorbar(means, y, xerr=np.vstack([means - lows, highs - means]), fmt="none", ecolor=PALETTE["dark"], capsize=2.5, lw=0.8)
    ax_d.set_yticks(y, aux_labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Mean value")
    ax_d.set_title("Auxiliary bias and balance diagnostics", loc="left", fontweight="bold", pad=3)
    ax_d.grid(axis="x", color="#E6E9ED", lw=0.7)
    add_panel_label(ax_d, "D")

    fig.suptitle(
        "Baseline diagnostic checks",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["dark"],
    )
    save_figure(fig, "figure_s1_baseline_diagnostics")
    plt.close(fig)


def main():
    configure_style()
    summary_path = BASELINE_DIR / "summary.csv"
    time_series_path = BASELINE_DIR / "time_series.csv"
    if not summary_path.exists() or not time_series_path.exists():
        raise FileNotFoundError("Baseline summary.csv and time_series.csv must exist before plotting.")

    summary = pd.read_csv(summary_path)
    time_series = pd.read_csv(time_series_path)
    make_main_figure(summary, time_series)
    make_diagnostic_figure(summary, time_series)
    print(f"Saved baseline figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
