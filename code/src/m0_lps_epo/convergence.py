from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def find_convergence_time(df: pd.DataFrame, W: int, epsilon: float):
    variables = ["VRB_P", "ParticipationRate", "H_mean", "B_algorithm_P"]
    n = len(df)
    if n < 2 * W:
        return None

    for idx in range(2 * W - 1, n):
        ok = True
        for var in variables:
            if var not in df.columns:
                continue
            current = np.nanmean(df[var].iloc[idx - W + 1 : idx + 1])
            previous = np.nanmean(df[var].iloc[idx - 2 * W + 1 : idx - W + 1])
            if not np.isfinite(current) or not np.isfinite(previous) or abs(current - previous) >= epsilon:
                ok = False
                break
        if ok:
            return int(df["t"].iloc[idx])
    return None


def stable_average(df: pd.DataFrame, t_star: Optional[int], W: int):
    if t_star is None:
        stable_df = df.tail(W).copy()
        converged = False
        stable_window = "terminal_window"
    else:
        stable_df = df[df["t"] >= t_star].copy()
        converged = True
        stable_window = "post_convergence"

    out = {
        "converged": converged,
        "t_star": t_star if t_star is not None else np.nan,
        "stable_window": stable_window,
    }
    for col in stable_df.columns:
        if col == "t":
            continue
        if np.issubdtype(stable_df[col].dtype, np.number):
            values = stable_df[col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            out[col] = float(np.mean(finite)) if len(finite) else np.nan
    return out
