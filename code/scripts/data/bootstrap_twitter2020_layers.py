# 说明：主样本分析不设置最低置信度；高置信度分析通过
# --min-confidence 0.75 指定，不再需要手动修改代码。
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from build_twitter2020_llm_layers import add_author_key, add_author_uniform_weights


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "empirical"
    / "twitter2020"
    / "llm_tweetkey"
    / "twitter2020_llm_labeled_tweets_analysis.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "empirical" / "twitter2020" / "cluster_bootstrap"
)

METRICS = [
    "mu",
    "P",
    "BPW",
    "U_mu",
    "CON",
    "moderate_share",
    "L_deltaB",
    "R_deltaB",
    "S_deltaB",
    "BAL_deltaB",
    "BTM",
]

LAYER_NAMES = [
    "F_Y_author_uniform",
    "F_P_post_uniform",
    "F_V_engagement_proxy",
    "F_V_retweet_proxy",
    "F_V_retweet_winsor99_proxy",
    "F_V_retweet_log1p_proxy",
    "F_V_retweet_drop_top0p1_proxy",
    "F_V_like_proxy",
]

RETWEET_TAIL_SENSITIVITY_SPECS = [
    (
        "F_V_retweet_proxy",
        "Main retweet weight",
        r"$1 + \mathrm{RT}$",
    ),
    (
        "F_V_retweet_winsor99_proxy",
        "99% winsorized retweet weight",
        r"$1 + \min(\mathrm{RT}, Q_{0.99})$",
    ),
    (
        "F_V_retweet_log1p_proxy",
        "Log-compressed retweet weight",
        r"$1 + \log(1 + \mathrm{RT})$",
    ),
    (
        "F_V_retweet_drop_top0p1_proxy",
        "Top 0.1% retweets removed",
        r"$1 + \mathrm{RT}$ after removing the top 0.1%",
    ),
]


@dataclass
class BootstrapData:
    n_authors: int
    score_values: np.ndarray
    layer_matrices_transposed: dict[str, sparse.csr_matrix]


def normalize_stance_labels(labels: list[str]) -> list[str]:
    """Normalize repeated CLI label filters while preserving their order."""
    normalized = []
    for label in labels:
        clean = str(label).strip().lower()
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def filtered_output_dir(
    min_confidence: float | None,
    excluded_stance_labels: list[str],
) -> Path:
    """Build a non-overlapping output directory for each sensitivity sample."""
    name = "cluster_bootstrap"
    if min_confidence is not None:
        threshold = f"{min_confidence:.2f}".replace(".", "")
        name += f"_confidence_ge_{threshold}"
    if excluded_stance_labels:
        safe_labels = "_and_".join(
            label.replace("-", "_").replace(" ", "_")
            for label in excluded_stance_labels
        )
        name += f"_exclude_{safe_labels}"
    return DEFAULT_OUTPUT_DIR.parent / name


def summarize_excluded_stance_labels(
    df: pd.DataFrame,
    excluded_stance_labels: list[str],
) -> pd.DataFrame:
    """Measure each excluded label's mass before applying the stance filter."""
    columns = [
        "stance_label",
        "tweet_count",
        "tweet_share",
        "authors_with_label",
        "author_share_with_label",
        "F_Y_weight_share",
        "F_P_weight_share",
        "F_V_RT_weight_share",
    ]
    if not excluded_stance_labels:
        return pd.DataFrame(columns=columns)
    if "stance_label" not in df.columns:
        raise ValueError("Input analysis file has no stance_label column.")

    working = df.copy()
    if "author_key" not in working.columns:
        working = add_author_key(working)
    working = add_author_uniform_weights(working)
    normalized_labels = working["stance_label"].astype("string").str.strip().str.lower()
    available_labels = set(normalized_labels.dropna().unique())
    missing_labels = [
        label for label in excluded_stance_labels if label not in available_labels
    ]
    if missing_labels:
        raise ValueError(
            "Requested excluded stance labels are absent from the analysis sample: "
            f"{missing_labels}"
        )
    author_count = working["author_key"].nunique()
    weight_columns = {
        "F_Y_weight_share": "author_uniform_weight",
        "F_P_weight_share": "post_weight",
        "F_V_RT_weight_share": "retweet_weight",
    }
    totals = {
        output_name: pd.to_numeric(working[column], errors="coerce").fillna(0).sum()
        for output_name, column in weight_columns.items()
    }

    rows = []
    for label in excluded_stance_labels:
        selected = normalized_labels.eq(label).fillna(False)
        authors_with_label = working.loc[selected, "author_key"].nunique()
        row = {
            "stance_label": label,
            "tweet_count": int(selected.sum()),
            "tweet_share": float(selected.mean()),
            "authors_with_label": int(authors_with_label),
            "author_share_with_label": float(authors_with_label / author_count),
        }
        for output_name, column in weight_columns.items():
            weights = pd.to_numeric(working[column], errors="coerce").fillna(0)
            total = totals[output_name]
            row[output_name] = (
                float(weights.loc[selected].sum() / total) if total > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summarize_sorted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
    delta_m: float,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return {metric: np.nan for metric in METRICS}

    total = weights.sum()
    normalized = weights / total
    mu = float(np.sum(normalized * values))

    cumulative_weight = np.cumsum(weights)
    cumulative_weighted_value = np.cumsum(weights * values)
    pairwise_numerator = np.sum(
        weights[1:]
        * (
            values[1:] * cumulative_weight[:-1]
            - cumulative_weighted_value[:-1]
        )
    )
    pairwise = float(2.0 * pairwise_numerator / total**2)

    # For p=1, BPW = 1 - 2 W1(F, 0.5 delta_0 + 0.5 delta_1).
    cumulative_normalized = np.cumsum(normalized)
    lower = cumulative_normalized - normalized
    upper = cumulative_normalized
    left_mass = np.clip(np.minimum(upper, 0.5) - lower, 0.0, None)
    right_mass = np.clip(upper - np.maximum(lower, 0.5), 0.0, None)
    wasserstein = float(
        np.sum(left_mass * values) + np.sum(right_mass * (1.0 - values))
    )
    bpw = float(np.clip(1.0 - 2.0 * wasserstein, 0.0, 1.0))

    left = float(np.sum(normalized * (values < 0.5 - delta_m)))
    right = float(np.sum(normalized * (values > 0.5 + delta_m)))
    side_mass = left + right
    balance = 0.0 if side_mass <= 0 else float(2.0 * min(left, right) / side_mass)
    return {
        "mu": mu,
        "P": pairwise,
        "BPW": bpw,
        "U_mu": float(2.0 * abs(mu - 0.5)),
        "CON": float(1.0 - 2.0 * pairwise),
        "moderate_share": float(
            np.sum(normalized * (np.abs(values - 0.5) < delta_m))
        ),
        "L_deltaB": left,
        "R_deltaB": right,
        "S_deltaB": side_mass,
        "BAL_deltaB": balance,
        "BTM": float(2.0 * min(left, right)),
    }


def prepare_bootstrap_data(df: pd.DataFrame) -> tuple[BootstrapData, pd.DataFrame]:
    required = {
        "tweet_key",
        "stance_score",
        "post_weight",
        "engagement_weight",
        "retweet_count",
        "retweet_weight",
        "like_weight",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input analysis file is missing columns: {sorted(missing)}")
    if "author_key" not in df.columns:
        df = add_author_key(df)

    df = df.loc[df["stance_score"].between(0, 1)].copy()
    retweets = (
        pd.to_numeric(df["retweet_count"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    winsor99_cutoff = float(retweets.quantile(0.99))
    drop_top0p1_cutoff = float(retweets.quantile(0.999))
    retweet_values = retweets.to_numpy(dtype=float)
    top_0p1_count = max(1, int(np.ceil(0.001 * len(df))))
    top_1_count = max(1, int(np.ceil(0.01 * len(df))))
    retweet_rank = np.argsort(retweet_values, kind="stable")
    drop_top0p1_mask = np.zeros(len(df), dtype=bool)
    drop_top0p1_mask[retweet_rank[-top_0p1_count:]] = True
    base_retweet_weight = 1.0 + retweet_values

    # These alternatives attenuate viral-tweet leverage while preserving a
    # one-unit baseline visibility mass for tweets with no retweets.
    df["retweet_weight_winsor99"] = 1.0 + retweets.clip(upper=winsor99_cutoff)
    df["retweet_weight_log1p"] = 1.0 + np.log1p(retweets)
    df["retweet_weight_drop_top0p1"] = np.where(
        drop_top0p1_mask,
        0.0,
        base_retweet_weight,
    )
    # Recompute after any confidence filter so each retained author has unit
    # total mass in F_Y, regardless of how many retained tweets they authored.
    df = add_author_uniform_weights(df)
    author_codes, author_keys = pd.factorize(df["author_key"], sort=True)
    score_values = np.sort(df["stance_score"].dropna().unique().astype(float))
    score_lookup = {value: index for index, value in enumerate(score_values)}
    score_codes = df["stance_score"].map(score_lookup).to_numpy(dtype=int)
    n_authors = len(author_keys)
    shape = (n_authors, len(score_values))

    matrices = {}
    layer_weight_columns = {
        "F_Y_author_uniform": "author_uniform_weight",
        "F_P_post_uniform": "post_weight",
        "F_V_engagement_proxy": "engagement_weight",
        "F_V_retweet_proxy": "retweet_weight",
        "F_V_retweet_winsor99_proxy": "retweet_weight_winsor99",
        "F_V_retweet_log1p_proxy": "retweet_weight_log1p",
        "F_V_retweet_drop_top0p1_proxy": "retweet_weight_drop_top0p1",
        "F_V_like_proxy": "like_weight",
    }
    for layer_name, column in layer_weight_columns.items():
        data = pd.to_numeric(df[column], errors="coerce").fillna(0).to_numpy(float)
        matrix = sparse.coo_matrix(
            (data, (author_codes, score_codes)),
            shape=shape,
            dtype=float,
        ).tocsr()
        matrices[layer_name] = matrix.T.tocsr()

    diagnostics = pd.DataFrame(
        [
            {
                "analysis_rows": len(df),
                "unique_authors": n_authors,
                "unique_stance_scores": len(score_values),
                "screen_name_author_rate": float(
                    df.get("author_key_source", pd.Series("", index=df.index))
                    .eq("screen_name")
                    .mean()
                ),
                "tweets_per_author_mean": float(len(df) / n_authors),
                "tweets_per_author_median": float(
                    df.groupby("author_key").size().median()
                ),
                "tweets_per_author_max": int(df.groupby("author_key").size().max()),
                "F_Y_author_weight_sum": float(df["author_uniform_weight"].sum()),
                "retweet_q99": winsor99_cutoff,
                "retweet_q999": drop_top0p1_cutoff,
                "retweet_max": float(retweets.max()),
                "retweet_weight_share_top_0p1": float(
                    base_retweet_weight[retweet_rank[-top_0p1_count:]].sum()
                    / base_retweet_weight.sum()
                ),
                "retweet_weight_share_top_1": float(
                    base_retweet_weight[retweet_rank[-top_1_count:]].sum()
                    / base_retweet_weight.sum()
                ),
                "retweet_weight_share_largest_tweet": float(
                    base_retweet_weight.max() / base_retweet_weight.sum()
                ),
                "tweets_removed_top_0p1": top_0p1_count,
                "share_removed_top_0p1": float(top_0p1_count / len(df)),
            }
        ]
    )
    return (
        BootstrapData(
            n_authors=n_authors,
            score_values=score_values,
            layer_matrices_transposed=matrices,
        ),
        diagnostics,
    )


def point_estimates(data: BootstrapData, delta_m: float) -> pd.DataFrame:
    rows = []
    ones = np.ones(data.n_authors, dtype=float)
    for layer_name, matrix_transposed in data.layer_matrices_transposed.items():
        score_weights = np.asarray(matrix_transposed.dot(ones)).ravel()
        rows.append(
            {
                "layer": layer_name,
                **summarize_sorted_distribution(
                    data.score_values,
                    score_weights,
                    delta_m,
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_cluster_counts(
    rng: np.random.Generator,
    n_authors: int,
    method: str,
) -> np.ndarray:
    if method == "multinomial":
        sampled = rng.integers(0, n_authors, size=n_authors)
        return np.bincount(sampled, minlength=n_authors).astype(float)
    if method == "poisson":
        counts = rng.poisson(1.0, size=n_authors).astype(float)
        while counts.sum() == 0:
            counts = rng.poisson(1.0, size=n_authors).astype(float)
        return counts
    raise ValueError(f"Unknown bootstrap method: {method}")


def run_chunk(
    data: BootstrapData,
    replication_ids: list[int],
    child_seeds: list[np.random.SeedSequence],
    delta_m: float,
    method: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for replication_id, child_seed in zip(replication_ids, child_seeds):
        rng = np.random.default_rng(child_seed)
        counts = draw_cluster_counts(rng, data.n_authors, method)
        for layer_name, matrix_transposed in data.layer_matrices_transposed.items():
            score_weights = np.asarray(matrix_transposed.dot(counts)).ravel()
            rows.append(
                {
                    "replication": replication_id,
                    "layer": layer_name,
                    **summarize_sorted_distribution(
                        data.score_values,
                        score_weights,
                        delta_m,
                    ),
                }
            )
    return rows


def summarize_layers(
    point: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for layer_name in LAYER_NAMES:
        point_row = point.loc[point["layer"].eq(layer_name)].iloc[0]
        boot_layer = replicates.loc[replicates["layer"].eq(layer_name)]
        for metric in METRICS:
            values = boot_layer[metric].dropna()
            rows.append(
                {
                    "layer": layer_name,
                    "metric": metric,
                    "estimate": point_row[metric],
                    "bootstrap_mean": values.mean(),
                    "bootstrap_se": values.std(ddof=1),
                    "ci95_low": values.quantile(0.025),
                    "ci95_high": values.quantile(0.975),
                    "bootstrap_replications_valid": len(values),
                }
            )
    return pd.DataFrame(rows)


def summarize_differences(
    point: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    contrasts = [
        ("F_P_minus_F_Y", "F_P_post_uniform", "F_Y_author_uniform"),
        ("F_V_engagement_minus_F_P", "F_V_engagement_proxy", "F_P_post_uniform"),
        ("F_V_retweet_minus_F_P", "F_V_retweet_proxy", "F_P_post_uniform"),
        (
            "F_V_retweet_winsor99_minus_F_P",
            "F_V_retweet_winsor99_proxy",
            "F_P_post_uniform",
        ),
        (
            "F_V_retweet_log1p_minus_F_P",
            "F_V_retweet_log1p_proxy",
            "F_P_post_uniform",
        ),
        (
            "F_V_retweet_drop_top0p1_minus_F_P",
            "F_V_retweet_drop_top0p1_proxy",
            "F_P_post_uniform",
        ),
        ("F_V_like_minus_F_P", "F_V_like_proxy", "F_P_post_uniform"),
        ("F_V_engagement_minus_F_Y", "F_V_engagement_proxy", "F_Y_author_uniform"),
    ]
    point_indexed = point.set_index("layer")
    indexed = replicates.set_index(["replication", "layer"])
    rows = []
    for contrast_name, upper, lower in contrasts:
        for metric in METRICS:
            point_difference = float(
                point_indexed.loc[upper, metric] - point_indexed.loc[lower, metric]
            )
            upper_values = indexed.xs(upper, level="layer")[metric]
            lower_values = indexed.xs(lower, level="layer")[metric]
            differences = (upper_values - lower_values).dropna()
            probability_positive = float(np.mean(differences > 0))
            rows.append(
                {
                    "contrast": contrast_name,
                    "metric": metric,
                    "estimate": point_difference,
                    "bootstrap_mean": differences.mean(),
                    "bootstrap_se": differences.std(ddof=1),
                    "ci95_low": differences.quantile(0.025),
                    "ci95_high": differences.quantile(0.975),
                    "probability_positive": probability_positive,
                    "two_sided_sign_p": min(
                        1.0,
                        2.0 * min(probability_positive, 1.0 - probability_positive),
                    ),
                    "bootstrap_replications_valid": len(differences),
                }
            )
    return pd.DataFrame(rows)


def summarize_retweet_tail_sensitivity(
    layer_summary: pd.DataFrame,
    differences: pd.DataFrame,
) -> pd.DataFrame:
    """Create an appendix-ready comparison of retweet-tail specifications."""
    layer_indexed = layer_summary.set_index(["layer", "metric"])
    contrast_indexed = differences.set_index(["contrast", "metric"])
    rows = []
    for layer, label, formula in RETWEET_TAIL_SENSITIVITY_SPECS:
        contrast = layer.replace("_proxy", "") + "_minus_F_P"
        row = {
            "layer": layer,
            "specification": label,
            "weight_formula": formula,
        }
        for metric in ("P", "BPW"):
            layer_stats = layer_indexed.loc[(layer, metric)]
            contrast_stats = contrast_indexed.loc[(contrast, metric)]
            row[f"{metric}_F_V"] = layer_stats["estimate"]
            row[f"{metric}_F_V_ci95_low"] = layer_stats["ci95_low"]
            row[f"{metric}_F_V_ci95_high"] = layer_stats["ci95_high"]
            row[f"delta_{metric}_vs_F_P"] = contrast_stats["estimate"]
            row[f"delta_{metric}_ci95_low"] = contrast_stats["ci95_low"]
            row[f"delta_{metric}_ci95_high"] = contrast_stats["ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Author-cluster bootstrap for Twitter 2020 empirical opinion layers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replications", type=int, default=5000)
    parser.add_argument(
        "--method",
        choices=["multinomial", "poisson"],
        default="multinomial",
        help="Multinomial is the ordinary author bootstrap; Poisson is a faster cluster-weight bootstrap.",
    )
    parser.add_argument("--delta-m", type=float, default=0.15)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum DeepSeek confidence for a sensitivity analysis.",
    )
    parser.add_argument(
        "--exclude-stance-label",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Exclude one stance label before constructing F_Y, F_P, and F_V. "
            "Repeat the option to exclude multiple labels."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    df = pd.read_csv(
        args.input,
        dtype={
            "tweet_key": "string",
            "user_id": "string",
            "user_screen_name": "string",
            "author_key": "string",
        },
    )
    input_rows = len(df)
    excluded_stance_labels = normalize_stance_labels(args.exclude_stance_label)
    if args.min_confidence is not None:
        if not 0.0 <= args.min_confidence <= 1.0:
            raise ValueError("--min-confidence must be between 0 and 1.")
        if "confidence" not in df.columns:
            raise ValueError("Input analysis file has no confidence column.")
        confidence = pd.to_numeric(df["confidence"], errors="coerce")
        df = df.loc[confidence.ge(args.min_confidence)].copy()
    rows_after_confidence_filter = len(df)
    if df.empty:
        raise ValueError("No rows remain after applying the confidence filter.")
    exclusion_diagnostics = summarize_excluded_stance_labels(
        df,
        excluded_stance_labels,
    )
    rows_before_stance_filter = len(df)
    if excluded_stance_labels:
        if "stance_label" not in df.columns:
            raise ValueError("Input analysis file has no stance_label column.")
        normalized_labels = df["stance_label"].astype("string").str.strip().str.lower()
        df = df.loc[~normalized_labels.isin(excluded_stance_labels)].copy()
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = filtered_output_dir(
            args.min_confidence,
            excluded_stance_labels,
        )
    if df.empty:
        raise ValueError("No rows remain after applying the requested sample filters.")
    data, diagnostics = prepare_bootstrap_data(df)

    point = point_estimates(data, args.delta_m)
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(args.replications)
    replication_ids = list(range(1, args.replications + 1))
    chunks = [
        (
            replication_ids[start : start + args.chunk_size],
            child_seeds[start : start + args.chunk_size],
        )
        for start in range(0, args.replications, args.chunk_size)
    ]

    all_rows = []
    completed = 0
    if args.max_workers <= 1:
        for ids, seeds in chunks:
            all_rows.extend(run_chunk(data, ids, seeds, args.delta_m, args.method))
            completed += len(ids)
            if args.show_progress:
                print(f"\rBootstrap: {completed}/{args.replications}", end="", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_chunk,
                    data,
                    ids,
                    seeds,
                    args.delta_m,
                    args.method,
                ): len(ids)
                for ids, seeds in chunks
            }
            for future in as_completed(futures):
                all_rows.extend(future.result())
                completed += futures[future]
                if args.show_progress:
                    print(f"\rBootstrap: {completed}/{args.replications}", end="", flush=True)
    if args.show_progress:
        print()

    replicates = pd.DataFrame(all_rows).sort_values(["replication", "layer"])
    layer_summary = summarize_layers(point, replicates)
    differences = summarize_differences(point, replicates)
    retweet_tail_sensitivity = summarize_retweet_tail_sensitivity(
        layer_summary,
        differences,
    )
    elapsed = time.perf_counter() - started
    diagnostics["bootstrap_method"] = args.method
    diagnostics["bootstrap_replications"] = args.replications
    diagnostics["input_rows_before_confidence_filter"] = input_rows
    diagnostics["minimum_confidence"] = args.min_confidence
    diagnostics["rows_after_confidence_filter"] = rows_after_confidence_filter
    diagnostics["confidence_filter_retention_rate"] = (
        rows_after_confidence_filter / input_rows
    )
    diagnostics["excluded_stance_labels"] = ";".join(excluded_stance_labels)
    diagnostics["rows_before_stance_filter"] = rows_before_stance_filter
    diagnostics["rows_excluded_by_stance_filter"] = rows_before_stance_filter - len(df)
    diagnostics["stance_filter_retention_rate"] = len(df) / rows_before_stance_filter
    diagnostics["seed"] = args.seed
    diagnostics["elapsed_seconds"] = elapsed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point.to_csv(
        args.output_dir / "twitter2020_layer_point_estimates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    replicates.to_csv(
        args.output_dir / "twitter2020_cluster_bootstrap_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    layer_summary.to_csv(
        args.output_dir / "twitter2020_cluster_bootstrap_layer_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    differences.to_csv(
        args.output_dir / "twitter2020_cluster_bootstrap_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    retweet_tail_sensitivity.to_csv(
        args.output_dir / "twitter2020_retweet_tail_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        args.output_dir / "twitter2020_cluster_bootstrap_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    exclusion_diagnostics.to_csv(
        args.output_dir / "twitter2020_excluded_stance_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Output directory: {args.output_dir}")
    print(f"Elapsed: {elapsed / 60:.2f} minutes")
    if excluded_stance_labels:
        print("\nExcluded stance-label diagnostics (before exclusion):")
        print(exclusion_diagnostics.to_string(index=False))
    print("\nMain P contrasts:")
    print(differences.loc[differences["metric"].eq("P")].to_string(index=False))
    print("\nRetweet-tail sensitivity (P and BPW):")
    print(retweet_tail_sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
