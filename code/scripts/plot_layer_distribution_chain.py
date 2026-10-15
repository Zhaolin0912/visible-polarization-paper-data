from pathlib import Path
import argparse
import sys
from matplotlib.ticker import FormatStrFormatter

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m0_lps_epo.config import load_config
from m0_lps_epo.model import run_single_model
from m0_lps_epo.rng import make_rng_streams
from m0_lps_epo.replications import run_replication


BASELINE_DIR = ROOT / "outputs" / "baseline"
OUTPUT_DIR = BASELINE_DIR / "layer_distributions"
BIN_EDGES = np.linspace(0.0, 1.0, 51)
LAYER_CSV_PATH = OUTPUT_DIR / "baseline_layer_distribution_chain.csv"
METADATA_PATH = OUTPUT_DIR / "layer_distribution_metadata.txt"

LAYER_INFO = [
    ("X", r"$F_X$", "Private opinions", "#8AB8E2"),
    ("A", r"$F_A$", "Participants' private opinions", "#60B071"),
    ("Y", r"$F_Y$", "Public expressions", "#F3A36D"),
    ("P", r"$F_P$", "Post-weighted public expressions", "#CA6464"),
    ("R", r"$F_R$", "Random structural reach", "#B4ADD7"),
    ("V", r"$F_V$", "Algorithmic visibility", "#81BDB9"),
]

RIDGELINE_SMOOTH_WINDOW = 5
PANEL_LABEL_X = 0.5
PANEL_LABEL_Y_TOP = -0.08
PANEL_LABEL_Y_BOTTOM = -0.28
DIFF_INFO = [
    ("A", "X", r"$F_A - F_X$", "Selection"),
    ("Y", "A", r"$F_Y - F_A$", "Expression"),
    ("P", "Y", r"$F_P - F_Y$", "Activity"),
    ("R", "P", r"$F_R - F_P$", "Reach"),
    ("V", "R", r"$F_V - F_R$", "Algorithm"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the six-layer distribution chain from cached data or by recomputing one representative replication."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Recompute the representative replication instead of reusing cached layer distribution data.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.labelsize": 14,
            "axes.titlesize": 14.5,
            "xtick.direction": "out",
            "ytick.direction": "in",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 330,
            "savefig.dpi": 600,
        }
    )


def save_pub(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg", "jpg"]:
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def write_small_multiples_panel_note():
    note_lines = [
        "Suggested panel-label mapping for the manuscript figure/caption:",
        "",
    ]
    for _layer_key, layer_label, long_title, _color in LAYER_INFO:
        note_lines.append(f"{layer_label}: {long_title}")
    (OUTPUT_DIR / "baseline_layer_distribution_chain_panel_labels.txt").write_text(
        "\n".join(note_lines) + "\n",
        encoding="utf-8",
    )


def smooth_series(values, window=RIDGELINE_SMOOTH_WINDOW):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def choose_representative_replication():
    summary = pd.read_csv(BASELINE_DIR / "summary.csv")
    target = float(summary["VRB_P"].mean())
    idx = (summary["VRB_P"] - target).abs().idxmin()
    row = summary.loc[idx]
    return int(row["replication_id"]), float(row["VRB_P"]), target


def collect_layer_distribution_df(replication_id: int):
    config = load_config(ROOT / "configs" / "baseline.yaml")
    first_pass = run_replication(config, replication_id=replication_id, variant="full")
    t_star = int(first_pass["summary"]["t_star"])

    rngs = make_rng_streams(config.experiment.seed, replication_id)
    second_pass = run_single_model(
        config=config,
        rngs=rngs,
        variant="full",
        collect_layer_distributions=True,
        layer_distribution_start=t_star,
        layer_distribution_bin_edges=BIN_EDGES,
    )
    layer_df = pd.DataFrame(second_pass["layer_distribution_records"])
    layer_df["replication_id"] = replication_id
    layer_df["t_star"] = t_star
    layer_df["T"] = config.simulation.T
    return layer_df, first_pass["summary"]


def plot_small_multiples_core(layer_df, label_mode="bottom"):
    if label_mode == "legend":
        figsize = (5, 3.5)
    else:
        figsize = (5.0, 3.5)
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    axes = axes.flatten()
    smoothed_max = 0.0
    for layer_key, _layer_label, _title, _color in LAYER_INFO:
        sub = layer_df[layer_df["layer"] == layer_key].sort_values("bin_mid")
        density_smooth = smooth_series(sub["mean_density"].to_numpy(), window=5)
        smoothed_max = max(smoothed_max, float(np.max(density_smooth)))
    ymax = 1.08 * smoothed_max

    for idx, (ax, (layer_key, layer_label, _long_title, color)) in enumerate(zip(axes, LAYER_INFO)):
        sub = layer_df[layer_df["layer"] == layer_key].sort_values("bin_mid")
        x = sub["bin_mid"].to_numpy()
        density_raw = sub["mean_density"].to_numpy()
        density_smooth = smooth_series(density_raw, window=5)
        ax.plot(x, density_raw, color=color, lw=0.8, alpha=0.25)
        ax.fill_between(x, density_smooth, color=color, alpha=0.24)
        ax.plot(x, density_smooth, color=color, lw=2.15, label=layer_label)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, ymax)
        ax.axvline(0.5, color="#C7CDD4", lw=0.8, ls="--", zorder=0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        if label_mode == "bottom":
            panel_label_y = PANEL_LABEL_Y_TOP if idx < 3 else PANEL_LABEL_Y_BOTTOM
            ax.text(
                PANEL_LABEL_X,
                panel_label_y,
                layer_label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10.5,
                color="#2F3A45",
                clip_on=False,
            )
        elif label_mode == "legend":
            ax.legend(
            frameon=False,
            loc="upper left",
            #ncol=3,
            #bbox_to_anchor=(0.5, 1),
            borderaxespad=0,
            handlelength=1,
            handletextpad=0.3,
            columnspacing=0.8,
            labelcolor="#2F3A45",
            )
            
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

    for ax in [axes[0], axes[3]]:
        ax.set_ylabel("Density",fontsize=14.5)
    fig.supxlabel("Opinion", ha="center", fontsize=14.5, fontfamily="Times New Roman")
    return fig


def plot_small_multiples(layer_df):
    return plot_small_multiples_core(layer_df, label_mode="bottom")


def plot_small_multiples_legend(layer_df):
    return plot_small_multiples_core(layer_df, label_mode="legend")


def plot_overlay(layer_df):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for layer_key, layer_label, _title, color in LAYER_INFO:
        sub = layer_df[layer_df["layer"] == layer_key].sort_values("bin_mid")
        ax.plot(sub["bin_mid"], sub["mean_density"], lw=2.0, color=color, label=layer_label)
    ax.axvline(0.5, color="#C7CDD4", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Average density")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    return fig


def plot_ridgeline(layer_df):
    fig, ax = plt.subplots(figsize=(8.7, 5.5), constrained_layout=True)
    ridge_gap = 1.05
    ridge_scale = 0.85
    label_x = -0.045

    ordered_layers = list(reversed(LAYER_INFO))
    for idx, (layer_key, layer_label, title, color) in enumerate(ordered_layers):
        sub = layer_df[layer_df["layer"] == layer_key].sort_values("bin_mid")
        x = sub["bin_mid"].to_numpy()
        density = smooth_series(sub["mean_density"].to_numpy())
        baseline = idx * ridge_gap
        ridge = baseline + ridge_scale * density

        ax.fill_between(x, baseline, ridge, color=color, alpha=0.32, linewidth=0.0, zorder=2 + idx)
        ax.plot(x, ridge, color=color, lw=2.0, zorder=3 + idx)
        ax.hlines(baseline, 0.0, 1.0, color="#E6E9ED", lw=0.8, zorder=1)
        ax.text(
            label_x,
            baseline + 0.12,
            f"{layer_label}: {title}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#2F3A45",
            clip_on=False,
        )

    ax.axvline(0.5, color="#C7CDD4", lw=0.9, ls="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ymax = (len(ordered_layers) - 1) * ridge_gap + ridge_scale * float(layer_df["mean_density"].max()) + 0.25
    ax.set_ylim(-0.15, ymax)
    ax.set_xlabel("Opinion")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    return fig


def plot_difference_panels(layer_df):
    layer_curves = {}
    for layer_key, _layer_label, _title, color in LAYER_INFO:
        sub = layer_df[layer_df["layer"] == layer_key].sort_values("bin_mid")
        layer_curves[layer_key] = {
            "x": sub["bin_mid"].to_numpy(),
            "density": smooth_series(sub["mean_density"].to_numpy()),
            "color": color,
        }

    fig, axes = plt.subplots(len(DIFF_INFO), 1, figsize=(8.6, 6.8), sharex=True, constrained_layout=True)
    diff_max = 0.0
    diff_cache = []
    for current, previous, label, mechanism in DIFF_INFO:
        x = layer_curves[current]["x"]
        diff = layer_curves[current]["density"] - layer_curves[previous]["density"]
        diff_cache.append((current, previous, label, mechanism, x, diff))
        diff_max = max(diff_max, float(np.max(np.abs(diff))))

    ylim = 1.08 * diff_max if diff_max > 0 else 0.1
    pos_fill = "#9DC3E6"
    neg_fill = "#D37F7F"
    line_color = "#2F3A45"

    for ax, (current, previous, label, mechanism, x, diff) in zip(axes, diff_cache):
        ax.axhline(0.0, color="#9AA3AD", lw=0.8, zorder=1)
        ax.axvline(0.5, color="#C7CDD4", lw=0.8, ls="--", zorder=0)
        ax.fill_between(x, 0.0, diff, where=diff >= 0.0, interpolate=True, color=pos_fill, alpha=0.55, zorder=2)
        ax.fill_between(x, 0.0, diff, where=diff <= 0.0, interpolate=True, color=neg_fill, alpha=0.55, zorder=2)
        ax.plot(x, diff, color=line_color, lw=1.8, zorder=3)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel("Delta density")
        ax.set_title(f"{label}: {mechanism}", loc="left", pad=3)
        ax.text(
            0.995,
            0.88,
            "More mass in later layer",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#4E79A7",
        )
        ax.text(
            0.995,
            0.12,
            "Less mass in later layer",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#B55D5D",
        )

    axes[-1].set_xlabel("Opinion")
    for ax in axes[:-1]:
        ax.label_outer()
    return fig


def write_metadata(replication_id, vrb_rep, vrb_mean, summary, layer_df):
    lines = [
        "Baseline layer distribution chain",
        f"Representative replication_id: {replication_id}",
        f"Representative replication VRB_P: {vrb_rep:.6f}",
        f"Baseline mean VRB_P across replications: {vrb_mean:.6f}",
        f"Representative replication t_star: {int(summary['t_star'])}",
        f"Layer averaging window start: {int(layer_df['t_star'].iloc[0])}",
        f"Layer averaging window end: {int(layer_df['T'].iloc[0])}",
        f"Window periods: {int(layer_df['window_periods'].iloc[0])}",
    ]
    METADATA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cached_metadata():
    if not METADATA_PATH.exists():
        return None
    meta = {}
    for line in METADATA_PATH.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def load_or_collect_layer_distribution_df(refresh_data: bool):
    if (not refresh_data) and LAYER_CSV_PATH.exists():
        layer_df = pd.read_csv(LAYER_CSV_PATH)
        meta = load_cached_metadata()
        if meta is not None:
            print(f"Loaded cached layer distribution data from: {LAYER_CSV_PATH}")
        else:
            print(f"Loaded cached layer distribution data from: {LAYER_CSV_PATH} (metadata file not found)")
        return layer_df, meta

    replication_id, rep_vrb, mean_vrb = choose_representative_replication()
    layer_df, summary = collect_layer_distribution_df(replication_id)
    layer_df.to_csv(LAYER_CSV_PATH, index=False)
    write_metadata(replication_id, rep_vrb, mean_vrb, summary, layer_df)
    print(f"Recomputed and saved layer distribution data to: {LAYER_CSV_PATH}")
    return layer_df, {
        "Representative replication_id": str(replication_id),
        "Representative replication VRB_P": f"{rep_vrb:.6f}",
        "Baseline mean VRB_P across replications": f"{mean_vrb:.6f}",
        "Representative replication t_star": str(int(summary["t_star"])),
        "Layer averaging window start": str(int(layer_df["t_star"].iloc[0])),
        "Layer averaging window end": str(int(layer_df["T"].iloc[0])),
        "Window periods": str(int(layer_df["window_periods"].iloc[0])),
    }


def main():
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    layer_df, metadata = load_or_collect_layer_distribution_df(refresh_data=args.refresh_data)

    fig_small = plot_small_multiples(layer_df)
    save_pub(fig_small, "baseline_layer_distribution_chain_small_multiples")

    fig_small_legend = plot_small_multiples_legend(layer_df)
    save_pub(fig_small_legend, "baseline_layer_distribution_chain_small_multiples_legend")

    write_small_multiples_panel_note()

    fig_overlay = plot_overlay(layer_df)
    save_pub(fig_overlay, "baseline_layer_distribution_chain_overlay")

    fig_ridgeline = plot_ridgeline(layer_df)
    save_pub(fig_ridgeline, "baseline_layer_distribution_chain_ridgeline")

    fig_diff = plot_difference_panels(layer_df)
    save_pub(fig_diff, "baseline_layer_distribution_chain_differences")

    print(f"Saved layer distribution figures to: {OUTPUT_DIR}")
    if metadata is not None:
        print(
            "Representative replication_id = "
            f"{metadata.get('Representative replication_id', 'NA')}, "
            f"VRB_P = {metadata.get('Representative replication VRB_P', 'NA')}, "
            f"baseline mean VRB_P = {metadata.get('Baseline mean VRB_P across replications', 'NA')}"
        )


if __name__ == "__main__":
    main()
