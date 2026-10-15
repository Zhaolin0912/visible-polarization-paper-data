from __future__ import annotations

import pandas as pd

from .replications import run_replication


def run_full_and_ablations(config, variants=None):
    if variants is None:
        # MatchedEqualRate removes rate heterogeneity while matching expected content supply each period.
        variants = [
            "random_selection",
            "honest_expression",
            "matched_equal_rate",
            "equal_posts",
            "random_visibility",
        ]

    all_summaries = []
    all_time_series = []
    full_counts_by_rep = {}
    full_post_counts_by_rep = {}

    for r in range(config.simulation.R):
        full_out = run_replication(config, r, variant="full")
        full_counts_by_rep[r] = full_out["participation_counts"]
        full_post_counts_by_rep[r] = full_out["post_counts"]
        all_summaries.append(full_out["summary"])
        ts = full_out["time_series"].copy()
        ts["replication_id"] = r
        ts["variant"] = "full"
        all_time_series.append(ts)

    for variant in variants:
        for r in range(config.simulation.R):
            full_counts = full_counts_by_rep[r] if variant == "random_selection" else None
            full_posts = full_post_counts_by_rep[r] if variant == "matched_equal_rate" else None
            out = run_replication(
                config,
                r,
                variant=variant,
                full_participation_counts=full_counts,
                full_post_counts=full_posts,
            )
            all_summaries.append(out["summary"])
            ts = out["time_series"].copy()
            ts["replication_id"] = r
            ts["variant"] = variant
            all_time_series.append(ts)

    return {
        "summary": pd.DataFrame(all_summaries),
        "time_series": pd.concat(all_time_series, ignore_index=True),
    }
