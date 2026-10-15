from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "results_candidate_figures"
SOURCE_DIR = FIGURE_DIR / "source_data"

THETA_DIR = ROOT / "outputs" / "thetaM_thetaC_sensitivity_full"
ALPHA_DIR = ROOT / "outputs" / "alpha_thetaM_sensitivity_full"
BETA_DIR = ROOT / "outputs" / "beta_sensitivity_full"
SIGMA_DIR = ROOT / "outputs" / "sigmaA_rhoKQ_sensitivity_full"

THETA_M_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0]
THETA_C_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25]
BETA_E_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
BETA_P_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
SIGMA_A_VALUES = [0.3, 0.5, 0.7, 0.9, 1.1]
RHO_KQ_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]

MECHANISM_COLUMNS = {
    "Selection": "B_selection_P",
    "Expression": "B_expression_P",
    "Activity": "B_activity_P",
    "Reach": "B_reach_P",
    "Algorithm": "B_algorithm_P",
}

MECHANISM_ABBREVIATIONS = {
    "Selection": "S",
    "Expression": "E",
    "Activity": "A",
    "Reach": "R",
    "Algorithm": "G",
}

DOMINANCE_ORDER = [
    "Weak bias",
    "Mixed",
    "Selection",
    "Expression",
    "Activity",
    "Reach",
    "Algorithm",
]

DOMINANCE_COLORS = {
    "Weak bias": "#ECECEC",
    "Mixed": "#CFB6DC",
    "Selection": "#9DC3E6",
    "Expression": "#FAD1A3",
    "Activity": "#D37F7F",
    "Reach": "#B48EAD",
    "Algorithm": "#74BA83",
}

SEQUENTIAL_CMAP = "YlOrRd"
POSNEG_CMAP = "RdBu_r"
def build_colormap(colors):
    return mcolors.LinearSegmentedColormap.from_list("custom_map", colors)


VRB_CMAP = build_colormap(["#FFF6C7", "#F5B487", "#C94F4F"])
VRB_SOFT_CMAP = build_colormap(["#FFF9D9", "#F6CEA6", "#DE8874"])
SELECTION_CMAP = build_colormap(["#EAF3FB", "#B7D4EE", "#5C90C8"])
# All observed expression contributions are negative. Map stronger compression
# to darker orange without reserving half of the color scale for unused positives.
#EXPRESSION_CMAP = build_colormap(["#C45A3C", "#F1A070", "#FCE8D8"])
EXPRESSION_CMAP = build_colormap(["#5C90C8", "#B7D4EE", "#EAF3FB"])
#ACTIVITY_CMAP = build_colormap(["#FAECE8", "#E6A091", "#B95D58"])
ACTIVITY_CMAP = build_colormap(["#EAF3FB", "#B7D4EE", "#5C90C8"])
CONCENTRATION_CMAP = build_colormap(["#EEF6F4", "#9BCBC4", "#3E8F88"])


def build_zero_anchored_colormap(vmin, vmax):
    """Use a linear numeric scale while anchoring the neutral color at zero."""
    if not vmin < 0.0 < vmax:
        raise ValueError("Zero-anchored colormap requires values on both sides of zero")
    zero_position = (0.0 - vmin) / (vmax - vmin)
    return mcolors.LinearSegmentedColormap.from_list(
        "algorithm_zero_anchored",
        [(0.0, "#5F8FAF"), (zero_position, "#F3F1E8"), (1.0, "#67A878")],
    )


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
            "axes.labelsize": 14.0,
            "axes.titlesize": 14.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "legend.fontsize": 11.5,
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
        path = str(FIGURE_DIR / f"{stem}.{suffix}")
        try:
            fig.savefig(path, bbox_inches="tight")
        except OSError:
            if suffix != "jpg":
                raise


def load_outputs(output_dir):
    report = pd.read_csv(output_dir / "report_mean.csv")
    summary = pd.read_csv(output_dir / "summary_by_replication.csv")
    return report, summary


def filter_parameter_grid(frame, x_col, y_col, x_values, y_values, frame_name, unique_cells):
    """Select the manuscript grid and fail loudly if any requested cell is incomplete."""
    filtered = frame.copy()
    filtered[x_col] = pd.to_numeric(filtered[x_col], errors="raise")
    filtered[y_col] = pd.to_numeric(filtered[y_col], errors="raise")
    filtered = filtered[
        filtered[x_col].isin(x_values) & filtered[y_col].isin(y_values)
    ].copy()

    expected = pd.MultiIndex.from_product(
        [y_values, x_values], names=[y_col, x_col]
    )
    observed = pd.MultiIndex.from_frame(filtered[[y_col, x_col]].drop_duplicates())
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"{frame_name} is missing parameter cells: {list(missing)}")

    if unique_cells and filtered.duplicated([y_col, x_col]).any():
        duplicates = filtered.loc[
            filtered.duplicated([y_col, x_col], keep=False), [y_col, x_col]
        ].drop_duplicates()
        raise ValueError(
            f"{frame_name} contains duplicate report rows: "
            f"{duplicates.to_dict(orient='records')}"
        )

    return filtered.sort_values([y_col, x_col]).reset_index(drop=True)


def load_theta_alpha_outputs():
    theta_report, theta_summary = load_outputs(THETA_DIR)
    alpha_report, alpha_summary = load_outputs(ALPHA_DIR)

    theta_report = filter_parameter_grid(
        theta_report,
        "theta_M",
        "theta_C",
        THETA_M_VALUES,
        THETA_C_VALUES,
        "theta_M x theta_C report",
        unique_cells=True,
    )
    theta_summary = filter_parameter_grid(
        theta_summary,
        "theta_M",
        "theta_C",
        THETA_M_VALUES,
        THETA_C_VALUES,
        "theta_M x theta_C replications",
        unique_cells=False,
    )
    alpha_report = filter_parameter_grid(
        alpha_report,
        "theta_M",
        "alpha",
        THETA_M_VALUES,
        ALPHA_VALUES,
        "alpha x theta_M report",
        unique_cells=True,
    )
    alpha_summary = filter_parameter_grid(
        alpha_summary,
        "theta_M",
        "alpha",
        THETA_M_VALUES,
        ALPHA_VALUES,
        "alpha x theta_M replications",
        unique_cells=False,
    )
    return theta_report, theta_summary, alpha_report, alpha_summary


def export_source(frame, name):
    frame.to_csv(SOURCE_DIR / name, index=False)


def make_numeric_grid(frame, x_col, y_col, value_col):
    pivot = (
        frame.pivot(index=y_col, columns=x_col, values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def cell_text_color(cmap_name, norm, value):
    r, g, b, _ = plt.get_cmap(cmap_name)(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def draw_numeric_heatmap(
    ax,
    frame,
    x_col,
    y_col,
    value_col,
    xlabel,
    ylabel,
    cmap,
    fmt="{:.3f}",
    center=None,
    show_xlabel=True,
    show_ylabel=True,
    cbar_tick_fmt=None,
    symmetric_center=True,
    cbar_ticks=None,
    aspect="auto",
):
    x_vals, y_vals, matrix = make_numeric_grid(frame, x_col, y_col, value_col)

    if center is None:
        norm = Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    elif not symmetric_center:
        norm = TwoSlopeNorm(
            vmin=float(np.nanmin(matrix)),
            vcenter=center,
            vmax=float(np.nanmax(matrix)),
        )
    else:
        vmax = float(np.nanmax(np.abs(matrix - center)))
        norm = TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)

    image = ax.imshow(matrix, origin="lower", aspect=aspect, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(xlabel if show_xlabel else "",fontsize=14)
    ax.set_ylabel(ylabel if show_ylabel else "",fontsize=14)
    #for row in range(matrix.shape[0]):
        #for col in range(matrix.shape[1]):
            #value = matrix[row, col]
            #ax.text(
                #col,
                #row,
                #fmt.format(value),
                #ha="center",
                #va="center",
                #fontsize=10.0,
                #color=cell_text_color(cmap, norm, value),
            #)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.12)
    cbar = ax.figure.colorbar(image, cax=cax)

    if cbar_tick_fmt is not None:
        # Set the formatter on the Colorbar itself; otherwise update_ticks()
        # will restore Matplotlib's default formatter and ignore the two-decimal request.
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        elif cbar_tick_fmt == "%.2f" and (norm.vmax - norm.vmin) <= 0.1:
            tick_start = np.ceil(norm.vmin * 100.0) / 100.0
            tick_end = np.floor(norm.vmax * 100.0) / 100.0
            ticks = np.arange(tick_start, tick_end + 0.005, 0.01)
            if len(ticks) >= 2:
                cbar.set_ticks(ticks)
            else:
                cbar.locator = MaxNLocator(nbins=5, min_n_ticks=4)
        else:
            cbar.locator = MaxNLocator(nbins=5, min_n_ticks=4)
        cbar.formatter = FormatStrFormatter(cbar_tick_fmt)
        cbar.update_ticks()
    
    cbar.ax.tick_params(labelsize=11.5)

#画三联图
def draw_mechanism_triptych(
    output_stem,
    frame,
    dominance_frame,
    x_col,
    y_col,
    mechanism_col,
    mechanism_label,
    x_label,
    y_label,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 3),
        gridspec_kw={"wspace": 0.5},
        constrained_layout=False,
    )

    mechanism_cmap = (
        SELECTION_CMAP if mechanism_col == "B_selection_P" else EXPRESSION_CMAP 
    )
    mechanism_center = None

    draw_numeric_heatmap(
        axes[0],
        frame,
        x_col=x_col,
        y_col=y_col,
        value_col=mechanism_col,
        xlabel=x_label,
        ylabel=y_label,
        cmap=mechanism_cmap,
        fmt="{:.3f}",
        center=mechanism_center,
        cbar_tick_fmt="%.3f",
    )
    draw_numeric_heatmap(
        axes[1],
        frame,
        x_col=x_col,
        y_col=y_col,
        value_col="VRB_P",
        xlabel=x_label,
        ylabel=y_label,
        cmap=VRB_CMAP,
        fmt="{:.3f}",
        show_ylabel=False,
        cbar_tick_fmt="%.3f",
    )
    draw_dominance_map(
        axes[2],
        dominance_frame,
        x_col=x_col,
        y_col=y_col,
        xlabel=x_label,
        ylabel=y_label,
    )
    axes[2].set_ylabel("")

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.98, wspace=0.3)
    save_figure(fig, output_stem)
    plt.close(fig)


def compute_dominance(summary_frame, x_col, y_col):
    grouped = summary_frame.groupby([y_col, x_col], as_index=False).mean(numeric_only=True)
    rows = []

    for _, row in grouped.iterrows():
        mechanism_values = {
            label: float(row[column])
            for label, column in MECHANISM_COLUMNS.items()
        }
        ranked = sorted(
            ((label, abs(value), value) for label, value in mechanism_values.items()),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_abs, top_signed = ranked[0]
        second_abs = ranked[1][1]
        p_x = float(row["P_X"])
        vrb_p = float(row["VRB_P"])
        tau_vrb = 0.05 * p_x
        tau_b = 0.02 * p_x

        # Match the manuscript rule: weak-bias first, then mixed regime, then dominant mechanism.
        if abs(vrb_p) < tau_vrb:
            category = "Weak bias"
            sign = ""
            mixed_pair = ""
        elif (top_abs - second_abs) < tau_b:
            category = "Mixed"
            sign = ""
            mixed_pair = (
                f"{MECHANISM_ABBREVIATIONS[top_label]}/"
                f"{MECHANISM_ABBREVIATIONS[ranked[1][0]]}"
            )
        else:
            category = top_label
            sign = "+" if top_signed > 0 else "-" if top_signed < 0 else "0"
            mixed_pair = ""

        rows.append(
            {
                x_col: float(row[x_col]),
                y_col: float(row[y_col]),
                "category": category,
                "sign": sign,
                "VRB_P": vrb_p,
                "P_X": p_x,
                "tau_VRB": tau_vrb,
                "tau_B": tau_b,
                "top_abs": top_abs,
                "second_abs": second_abs,
                "top_signed": top_signed,
                "mixed_pair": mixed_pair,
            }
        )

    return pd.DataFrame(rows)


def draw_dominance_map(ax, dominance_frame, x_col, y_col, xlabel, ylabel):
    plot_frame = dominance_frame.copy()
    plot_frame["category_code"] = plot_frame["category"].map(
        {label: idx for idx, label in enumerate(DOMINANCE_ORDER)}
    )

    pivot = (
        plot_frame.pivot(index=y_col, columns=x_col, values="category_code")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    sign_pivot = (
        plot_frame.pivot(index=y_col, columns=x_col, values="sign")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    pair_pivot = (
        plot_frame.pivot(index=y_col, columns=x_col, values="mixed_pair")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    matrix = pivot.to_numpy(dtype=float)
    sign_matrix = sign_pivot.to_numpy(dtype=object)
    pair_matrix = pair_pivot.to_numpy(dtype=object)

    cmap = ListedColormap([DOMINANCE_COLORS[label] for label in DOMINANCE_ORDER])
    ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(DOMINANCE_ORDER) - 0.5,
        alpha=0.8,
    )

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(xlabel,fontsize=14)
    ax.set_ylabel(ylabel,fontsize=14)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            category = DOMINANCE_ORDER[int(matrix[row_idx, col_idx])]
            sign = sign_matrix[row_idx, col_idx]
            if category == "Weak bias":
                text = "W"
            elif category == "Mixed":
                text = pair_matrix[row_idx, col_idx]
            else:
                text = f"{MECHANISM_ABBREVIATIONS[category]}{sign}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                fontweight="semibold",
            )

#2 × 2 图 VRB和主导图
def plot_theta_alpha_landscapes():
    theta_report, theta_summary, alpha_report, alpha_summary = load_theta_alpha_outputs()

    theta_dom = compute_dominance(theta_summary, "theta_M", "theta_C")
    alpha_dom = compute_dominance(alpha_summary, "theta_M", "alpha")

    export_source(
        theta_report[["theta_M", "theta_C", "VRB_P", "B_selection_P", "B_expression_P", "B_activity_P", "B_algorithm_P"]],
        "figure_parameter_thetaM_thetaC_heatmap_source.csv",
    )
    export_source(theta_dom, "figure_parameter_thetaM_thetaC_dominance_source.csv")
    export_source(
        alpha_report[["theta_M", "alpha", "VRB_P", "B_selection_P", "B_expression_P", "B_activity_P", "B_algorithm_P"]],
        "figure_parameter_alpha_thetaM_heatmap_source.csv",
    )
    export_source(alpha_dom, "figure_parameter_alpha_thetaM_dominance_source.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 4), constrained_layout=True)

    draw_numeric_heatmap(
        axes[0, 0],
        theta_report,
        x_col="theta_M",
        y_col="theta_C",
        value_col="VRB_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\theta_C$",
        cmap=SEQUENTIAL_CMAP,
        #title=r"$\theta_M \times \theta_C$: VRB-P landscape",
    )
    draw_dominance_map(
        axes[0, 1],
        theta_dom,
        x_col="theta_M",
        y_col="theta_C",
        xlabel=r"$\theta_M$",
        ylabel=r"$\theta_C$",
    )
    draw_numeric_heatmap(
        axes[1, 0],
        alpha_report,
        x_col="theta_M",
        y_col="alpha",
        value_col="VRB_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\alpha$",
        cmap=SEQUENTIAL_CMAP,
        #title=r"$\alpha \times \theta_M$: VRB-P landscape",
    )
    draw_dominance_map(
        axes[1, 1],
        alpha_dom,
        x_col="theta_M",
        y_col="alpha",
        xlabel=r"$\theta_M$",
        ylabel=r"$\alpha$",
    )

    save_figure(fig, "figure_parameter_landscapes_and_dominance")
    plt.close(fig)

#输出两张三联图
def plot_theta_alpha_mechanism_triptychs():
    theta_report, theta_summary, alpha_report, alpha_summary = load_theta_alpha_outputs()

    theta_dom = compute_dominance(theta_summary, "theta_M", "theta_C")
    alpha_dom = compute_dominance(alpha_summary, "theta_M", "alpha")

    draw_mechanism_triptych(
        output_stem="figure_thetaM_thetaC_mechanism_triptych",
        frame=theta_report,
        dominance_frame=theta_dom,
        x_col="theta_M",
        y_col="theta_C",
        mechanism_col="B_selection_P",
        mechanism_label=r"$B_{\mathrm{selection}}^{P}$",
        x_label=r"$\theta_M$",
        y_label=r"$\theta_C$",
    )

    draw_mechanism_triptych(
        output_stem="figure_alpha_thetaM_mechanism_triptych",
        frame=alpha_report,
        dominance_frame=alpha_dom,
        x_col="theta_M",
        y_col="alpha",
        mechanism_col="B_expression_P",
        mechanism_label=r"$B_{\mathrm{expression}}^{P}$",
        x_label=r"$\theta_M$",
        y_label=r"$\alpha$",
    )

#2 × 3 图
def plot_theta_alpha_mechanism_suite():
    theta_report, theta_summary, alpha_report, alpha_summary = load_theta_alpha_outputs()

    theta_dom = compute_dominance(theta_summary, "theta_M", "theta_C")
    alpha_dom = compute_dominance(alpha_summary, "theta_M", "alpha")

    export_source(
        theta_report[
            [
                "theta_M",
                "theta_C",
                "B_selection_P",
                "VRB_P",
                "B_expression_P",
                "B_activity_P",
                "B_algorithm_P",
            ]
        ],
        "figure_parameter_thetaM_thetaC_mechanism_suite_source.csv",
    )
    export_source(
        alpha_report[
            [
                "theta_M",
                "alpha",
                "B_expression_P",
                "VRB_P",
                "B_selection_P",
                "B_activity_P",
                "B_algorithm_P",
            ]
        ],
        "figure_parameter_alpha_thetaM_mechanism_suite_source.csv",
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 7),
        constrained_layout=True,
    )

    draw_numeric_heatmap(
        axes[0, 0],
        theta_report,
        x_col="theta_M",
        y_col="theta_C",
        value_col="B_selection_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\theta_C$",
        cmap=POSNEG_CMAP,
        fmt="{:.3f}",
        center=0.0,
    )
    draw_numeric_heatmap(
        axes[0, 1],
        theta_report,
        x_col="theta_M",
        y_col="theta_C",
        value_col="VRB_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\theta_C$",
        cmap=SEQUENTIAL_CMAP,
        fmt="{:.3f}",
        show_ylabel=False,
    )
    draw_dominance_map(
        axes[0, 2],
        theta_dom,
        x_col="theta_M",
        y_col="theta_C",
        xlabel=r"$\theta_M$",
        ylabel=r"$\theta_C$",
    )
    axes[0, 2].set_ylabel("")

    draw_numeric_heatmap(
        axes[1, 0],
        alpha_report,
        x_col="theta_M",
        y_col="alpha",
        value_col="B_expression_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\alpha$",
        cmap=POSNEG_CMAP,
        fmt="{:.3f}",
        center=0.0,
    )
    draw_numeric_heatmap(
        axes[1, 1],
        alpha_report,
        x_col="theta_M",
        y_col="alpha",
        value_col="VRB_P",
        xlabel=r"$\theta_M$",
        ylabel=r"$\alpha$",
        cmap=SEQUENTIAL_CMAP,
        fmt="{:.3f}",
        show_ylabel=False,
    )
    draw_dominance_map(
        axes[1, 2],
        alpha_dom,
        x_col="theta_M",
        y_col="alpha",
        xlabel=r"$\theta_M$",
        ylabel=r"$\alpha$",
    )
    axes[1, 2].set_ylabel("")

    #add_panel_metric_label(axes[0, 0], r"$B_{\mathrm{selection}}^{P}$")
    #add_panel_metric_label(axes[0, 1], "VRB-P")
    #add_panel_metric_label(axes[0, 2], "Dominance")
    #add_panel_metric_label(axes[1, 0], r"$B_{\mathrm{expression}}^{P}$")
    #add_panel_metric_label(axes[1, 1], "VRB-P")
    #add_panel_metric_label(axes[1, 2], "Dominance")

    save_figure(fig, "figure_parameter_mechanism_suite")
    plt.close(fig)

#算法参数热力图
def plot_beta_algorithm_landscapes():
    beta_report, _ = load_outputs(BETA_DIR)
    beta_report = filter_parameter_grid(
        beta_report,
        "beta_E",
        "beta_P",
        BETA_E_VALUES,
        BETA_P_VALUES,
        "beta_E x beta_P report",
        unique_cells=True,
    )
    algorithm_min = float(beta_report["B_algorithm_P"].min())
    algorithm_max = float(beta_report["B_algorithm_P"].max())
    algorithm_cmap = build_zero_anchored_colormap(algorithm_min, algorithm_max)
    export_source(
        beta_report[["beta_E", "beta_P", "VRB_P", "B_algorithm_P", "EPRB_actual"]],
        "figure_beta_algorithm_landscapes_source.csv",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
        constrained_layout=False,
    )

    draw_numeric_heatmap(
        axes[0],
        beta_report,
        x_col="beta_E",
        y_col="beta_P",
        value_col="B_algorithm_P",
        xlabel=r"$\beta_E$",
        ylabel=r"$\beta_P$",
        cmap=algorithm_cmap,
        fmt="{:.3f}",
        cbar_tick_fmt="%.3f",
        cbar_ticks=[-0.010, 0.000, 0.010, 0.020],
        aspect="equal",
    )
    draw_numeric_heatmap(
        axes[1],
        beta_report,
        x_col="beta_E",
        y_col="beta_P",
        value_col="VRB_P",
        xlabel=r"$\beta_E$",
        ylabel=r"$\beta_P$",
        cmap=VRB_SOFT_CMAP,
        fmt="{:.3f}",
        show_ylabel=False,
        cbar_tick_fmt="%.3f",
        cbar_ticks=[0.010, 0.020, 0.030, 0.040, 0.050],
        aspect="equal",
    )
    axes[1].tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.16, top=0.98, wspace=0.10)
    save_figure(fig, "figure_beta_algorithm_landscapes")
    plt.close(fig)

#活动机制参数热力图
def plot_sigma_activity_landscapes():
    sigma_report, _ = load_outputs(SIGMA_DIR)
    sigma_report = filter_parameter_grid(
        sigma_report,
        "sigma_A",
        "rho_kq",
        SIGMA_A_VALUES,
        RHO_KQ_VALUES,
        "sigma_A x rho_kq report",
        unique_cells=True,
    )
    export_source(
        sigma_report[["sigma_A", "rho_kq", "B_activity_P", "VMA_10pct", "Gini_V", "VRB_P"]],
        "figure_sigmaA_rhoKQ_activity_landscapes_source.csv",
    )

    # The three panels separate mechanism contribution, final bias, and exposure concentration.
    # Gini(V) remains in the exported source data as an alternative appendix diagnostic.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 4.5),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.22},
        constrained_layout=False,
    )

    draw_numeric_heatmap(
        axes[0],
        sigma_report,
        x_col="sigma_A",
        y_col="rho_kq",
        value_col="B_activity_P",
        xlabel=r"$\sigma_A$",
        ylabel=r"$\rho$",
        cmap=ACTIVITY_CMAP,
        fmt="{:.3f}",
        show_xlabel=True,
        show_ylabel=True,
        cbar_tick_fmt="%.3f",
        aspect="equal",
    )
    draw_numeric_heatmap(
        axes[1],
        sigma_report,
        x_col="sigma_A",
        y_col="rho_kq",
        value_col="VRB_P",
        xlabel=r"$\sigma_A$",
        ylabel=r"$\rho$",
        cmap=VRB_SOFT_CMAP,
        fmt="{:.3f}",
        show_xlabel=True,
        show_ylabel=False,
        cbar_tick_fmt="%.3f",
        aspect="equal",
    )
    draw_numeric_heatmap(
        axes[2],
        sigma_report,
        x_col="sigma_A",
        y_col="rho_kq",
        value_col="VMA_10pct",
        xlabel=r"$\sigma_A$",
        ylabel=r"$\rho$",
        cmap=CONCENTRATION_CMAP,
        fmt="{:.2f}",
        show_xlabel=True,
        show_ylabel=False,
        cbar_tick_fmt="%.1f",
        aspect="equal",
    )
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.985, wspace=0.22)
    save_figure(fig, "figure_sigmaA_rhoKQ_activity_landscapes")
    plt.close(fig)


def main():
    configure_style()
    ensure_dirs()
    plot_theta_alpha_landscapes()
    plot_theta_alpha_mechanism_triptychs()
    plot_theta_alpha_mechanism_suite()
    plot_beta_algorithm_landscapes()
    plot_sigma_activity_landscapes()
    print(f"Saved parameter-sensitivity figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
