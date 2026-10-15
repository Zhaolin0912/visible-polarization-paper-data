from __future__ import annotations

import numpy as np
import pandas as pd

#分几个区间
def default_bin_edges(Q: int) -> np.ndarray:
    if Q < 2:
        raise ValueError("transition_bins_Q must be at least 2.")
    return np.linspace(0.0, 1.0, Q + 1)

#把每个值映射到 bin index
def bin_indices(values, bin_edges):
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    return np.clip(idx, 0, len(bin_edges) - 2).astype(int)

#左闭右开
def bin_label(bin_id: int, bin_edges) -> str:
    left = bin_edges[bin_id]
    right = bin_edges[bin_id + 1]
    right_bracket = "]" if bin_id == len(bin_edges) - 2 else ")"
    return f"[{left:.2f},{right:.2f}{right_bracket}"


def transition_count_matrix(M, x, bin_edges):
    Q = len(bin_edges) - 1
    bins = bin_indices(x, bin_edges)
    counts = np.zeros((Q, Q), dtype=float)

    for observer, items in enumerate(M):
        observer_bin = bins[observer]
        for producer, _item_id in items:
            producer_bin = bins[producer]
            counts[producer_bin, observer_bin] += 1.0

    return counts


def transition_records_from_counts(counts, bin_edges, t: int, visibility: str):
    records = []
    column_totals = counts.sum(axis=0)
    Q = counts.shape[0]

    for producer_bin in range(Q):
        for observer_bin in range(Q):
            total = column_totals[observer_bin]
            prob = np.nan if total <= 0 else counts[producer_bin, observer_bin] / total
            records.append(
                {
                    "t": t,
                    "visibility": visibility,
                    "producer_bin": producer_bin,
                    "observer_bin": observer_bin,
                    "producer_bin_label": bin_label(producer_bin, bin_edges),
                    "observer_bin_label": bin_label(observer_bin, bin_edges),
                    "producer_bin_left": bin_edges[producer_bin],
                    "producer_bin_right": bin_edges[producer_bin + 1],
                    "observer_bin_left": bin_edges[observer_bin],
                    "observer_bin_right": bin_edges[observer_bin + 1],
                    "exposures": counts[producer_bin, observer_bin],
                    "observer_bin_exposures": total,
                    "prob": prob,
                    "valid_observer_bin": total > 0,
                }
            )

    return records


def transition_records_for_visibility(M, x, bin_edges, t: int, visibility: str):
    counts = transition_count_matrix(M, x, bin_edges)
    return transition_records_from_counts(counts, bin_edges, t=t, visibility=visibility)


def summarize_transition_records(records_df: pd.DataFrame, t_star, W: int):
    if records_df.empty:
        return pd.DataFrame()

    if t_star is None or not np.isfinite(t_star):
        t_min = max(int(records_df["t"].max()) - W + 1, int(records_df["t"].min()))
        stable_window = "terminal_window"
    else:
        t_min = int(t_star)
        stable_window = "post_convergence"

    stable = records_df[records_df["t"] >= t_min].copy()
    group_cols = [
        "visibility",
        "producer_bin",
        "observer_bin",
        "producer_bin_label",
        "observer_bin_label",
        "producer_bin_left",
        "producer_bin_right",
        "observer_bin_left",
        "observer_bin_right",
    ]
    out = stable.groupby(group_cols, as_index=False)["exposures"].sum()
    observer_totals = out.groupby(["visibility", "observer_bin"])["exposures"].transform("sum")
    out["observer_bin_exposures"] = observer_totals
    out["prob"] = np.where(observer_totals > 0, out["exposures"] / observer_totals, np.nan)
    out["valid_observer_bin"] = observer_totals > 0
    out["t_min"] = t_min
    out["t_max"] = int(stable["t"].max()) if len(stable) else np.nan
    out["stable_window"] = stable_window
    return out

