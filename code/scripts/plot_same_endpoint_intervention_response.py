from pathlib import Path
import argparse
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGURE_DIR = ROOT / "outputs" / "results_candidate_figures"

CONTROL_COLOR = "#9ECEA7"
INTERVENTION_COLOR = "#F4B488"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot matched-endpoint intervention responses.")
    parser.add_argument("--smoke-test", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "figure.dpi": 330,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["jpg", "png", "pdf", "svg"]:
        fig.savefig(FIGURE_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def plot_path(ax, series, condition, intervention_time, evaluation_start):
    subset = series[(series["condition"] == condition) & (series["metric"] == "VRB_P_actual")]
    for branch, color, label, linestyle in [
        ("control", CONTROL_COLOR, "Control", "-"),
        ("intervention", INTERVENTION_COLOR, "Random visibility", "--"),
    ]:
        rows = subset[subset["branch"] == branch].sort_values("t")
        ax.plot(rows["t"], rows["mean"], color=color, lw=1.8, ls=linestyle, label=label)
        ax.fill_between(
            rows["t"],
            rows["ci95_low"],
            rows["ci95_high"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.axvline(intervention_time, color="#6B7280", lw=1.0, ls=":")
    ax.axvspan(evaluation_start, subset["t"].max(), color="#D9D9D9", alpha=0.16, zorder=0)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\mathrm{VRB}_P$")
    ax.text(0.03, 0.96, f"Condition {condition}", transform=ax.transAxes, va="top", fontsize=11.5)
    ax.grid(axis="y", color="#E6E9ED", lw=0.8, ls="--")


def main():
    args = parse_args()
    configure_style()
    input_name = "same_endpoint_intervention_smoke" if args.smoke_test else "same_endpoint_intervention_full"
    input_dir = ROOT / "outputs" / input_name
    series_path = input_dir / "time_series_report.csv"
    design_path = input_dir / "experiment_design.csv"
    for path in [series_path, design_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    series = pd.read_csv(series_path)
    design = pd.read_csv(design_path)
    intervention_time = int(design["intervention_time"].iloc[0])
    evaluation_start = int(design["evaluation_start"].iloc[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.2, 3),
        sharey=True,
        constrained_layout=True,
    )
    plot_path(axes[0], series, "A", intervention_time, evaluation_start)
    plot_path(axes[1], series, "B", intervention_time, evaluation_start)
    axes[1].set_ylabel("")
    axes[0].legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        borderaxespad=0.0,
        prop={"family": "Times New Roman", "size": 10.5},
        handlelength=1.25,
        handletextpad=0.35,
    )
    stem = "figure_same_endpoint_intervention_response_smoke" if args.smoke_test else "figure_same_endpoint_intervention_response"
    save_figure(fig, stem)
    print(f"Saved intervention-response figure to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
