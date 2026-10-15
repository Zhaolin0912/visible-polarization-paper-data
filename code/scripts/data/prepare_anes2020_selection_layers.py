from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "anes2020" / "anes_timeseries_2020_csv_20220210.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "empirical" / "anes2020" / "selection_layers"

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
LAYERS = ["F_X", "F_A_online_political_participants"]


@dataclass
class PreparedData:
    sample: pd.DataFrame
    score_values: np.ndarray
    layer_cluster_weights: Dict[str, np.ndarray]
    clusters_by_stratum: List[np.ndarray]
    cluster_table: pd.DataFrame


def summarize_sorted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
    delta_m: float,
) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
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


def prepare_analysis_data(
    input_csv: Path,
    biden_col: str,
    trump_col: str,
    participation_col: str,
    participant_value: float,
    weight_col: str,
    psu_col: str,
    strata_col: str,
) -> Tuple[PreparedData, pd.DataFrame]:
    id_col = "V200001"
    required = [
        id_col,
        biden_col,
        trump_col,
        participation_col,
        weight_col,
        psu_col,
        strata_col,
    ]
    raw = pd.read_csv(input_csv, usecols=required, low_memory=False)
    renamed = raw.rename(
        columns={
            id_col: "respondent_id",
            biden_col: "ft_biden",
            trump_col: "ft_trump",
            participation_col: "online_political_posting",
            weight_col: "survey_weight",
            psu_col: "psu",
            strata_col: "stratum",
        }
    )
    numeric_columns = [
        "ft_biden",
        "ft_trump",
        "online_political_posting",
        "survey_weight",
        "psu",
        "stratum",
    ]
    for column in numeric_columns:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

    valid_thermometers = renamed["ft_biden"].between(0, 100) & renamed[
        "ft_trump"
    ].between(0, 100)
    valid_participation = renamed["online_political_posting"].isin([1, 2])
    valid_weight = np.isfinite(renamed["survey_weight"]) & (
        renamed["survey_weight"] > 0
    )
    valid_design = renamed["psu"].notna() & renamed["stratum"].notna()
    common_valid = (
        valid_thermometers & valid_participation & valid_weight & valid_design
    )

    sample = renamed.loc[common_valid].copy()
    sample["x"] = (
        sample["ft_trump"] - sample["ft_biden"] + 100.0
    ) / 200.0
    sample["x"] = sample["x"].clip(0.0, 1.0)
    sample["online_political_participant"] = sample[
        "online_political_posting"
    ].eq(participant_value)
    sample["weight_norm_fx"] = sample["survey_weight"] / sample[
        "survey_weight"
    ].sum()
    participant_weight_sum = sample.loc[
        sample["online_political_participant"], "survey_weight"
    ].sum()
    sample["weight_norm_fa"] = 0.0
    sample.loc[
        sample["online_political_participant"], "weight_norm_fa"
    ] = (
        sample.loc[
            sample["online_political_participant"], "survey_weight"
        ]
        / participant_weight_sum
    )
    sample["source"] = "ANES2020_general_online_political_participation"

    cluster_key = pd.MultiIndex.from_frame(sample[["stratum", "psu"]])
    cluster_codes, unique_clusters = pd.factorize(cluster_key, sort=True)
    sample["cluster_id"] = cluster_codes
    cluster_table = unique_clusters.to_frame(index=False)
    cluster_table.columns = ["stratum", "psu"]
    cluster_table.insert(0, "cluster_id", np.arange(len(cluster_table)))

    score_values = np.sort(sample["x"].unique().astype(float))
    score_lookup = {value: index for index, value in enumerate(score_values)}
    score_codes = sample["x"].map(score_lookup).to_numpy(dtype=int)
    n_scores = len(score_values)
    n_clusters = len(cluster_table)

    layer_cluster_weights = {}
    layer_masks = {
        "F_X": np.ones(len(sample), dtype=bool),
        "F_A_online_political_participants": sample[
            "online_political_participant"
        ].to_numpy(dtype=bool),
    }
    base_weights = sample["survey_weight"].to_numpy(dtype=float)
    for layer, layer_mask in layer_masks.items():
        matrix = np.zeros((n_scores, n_clusters), dtype=float)
        np.add.at(
            matrix,
            (score_codes[layer_mask], cluster_codes[layer_mask]),
            base_weights[layer_mask],
        )
        layer_cluster_weights[layer] = matrix

    clusters_by_stratum = [
        group["cluster_id"].to_numpy(dtype=int)
        for _, group in cluster_table.groupby("stratum", sort=True)
    ]

    diagnostics = pd.DataFrame(
        [
            {
                "input_csv": str(input_csv),
                "n_raw": len(raw),
                "n_common_valid": len(sample),
                "n_online_political_participants": int(
                    sample["online_political_participant"].sum()
                ),
                "unweighted_participation_rate": float(
                    sample["online_political_participant"].mean()
                ),
                "weighted_participation_rate": float(
                    participant_weight_sum / sample["survey_weight"].sum()
                ),
                "n_excluded_thermometers": int((~valid_thermometers).sum()),
                "n_excluded_participation": int((~valid_participation).sum()),
                "n_excluded_weight": int((~valid_weight).sum()),
                "n_excluded_design": int((~valid_design).sum()),
                "n_strata": len(clusters_by_stratum),
                "n_psu_stratum_pairs": n_clusters,
                "min_psus_per_stratum": min(map(len, clusters_by_stratum)),
                "max_psus_per_stratum": max(map(len, clusters_by_stratum)),
                "n_unique_opinion_scores": n_scores,
                "biden_col": biden_col,
                "trump_col": trump_col,
                "participation_col": participation_col,
                "participant_value": participant_value,
                "weight_col": weight_col,
                "psu_col": psu_col,
                "strata_col": strata_col,
            }
        ]
    )
    return (
        PreparedData(
            sample=sample,
            score_values=score_values,
            layer_cluster_weights=layer_cluster_weights,
            clusters_by_stratum=clusters_by_stratum,
            cluster_table=cluster_table,
        ),
        diagnostics,
    )


def draw_stratified_psu_counts(
    rng: np.random.Generator,
    clusters_by_stratum: List[np.ndarray],
    n_clusters: int,
    method: str,
) -> np.ndarray:
    counts = np.zeros(n_clusters, dtype=float)
    for cluster_ids in clusters_by_stratum:
        n_psus = len(cluster_ids)
        if n_psus <= 1:
            counts[cluster_ids] = 1.0
            continue
        if method == "rao_wu":
            draws = n_psus - 1
            sampled_positions = rng.integers(0, n_psus, size=draws)
            multiplicities = np.bincount(
                sampled_positions, minlength=n_psus
            ).astype(float)
            counts[cluster_ids] = multiplicities * n_psus / draws
        elif method == "ordinary":
            sampled_positions = rng.integers(0, n_psus, size=n_psus)
            counts[cluster_ids] = np.bincount(
                sampled_positions, minlength=n_psus
            ).astype(float)
        else:
            raise ValueError("Unknown bootstrap method: {0}".format(method))
    return counts


def point_estimates(data: PreparedData, delta_m: float) -> pd.DataFrame:
    ones = np.ones(len(data.cluster_table), dtype=float)
    rows = []
    for layer in LAYERS:
        score_weights = data.layer_cluster_weights[layer].dot(ones)
        rows.append(
            {
                "layer": layer,
                **summarize_sorted_distribution(
                    data.score_values, score_weights, delta_m
                ),
            }
        )
    return pd.DataFrame(rows)


def run_bootstrap(
    data: PreparedData,
    replications: int,
    seed: int,
    delta_m: float,
    method: str,
    progress_every: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    participation_rows = []
    started = time.perf_counter()
    for replication in range(1, replications + 1):
        cluster_counts = draw_stratified_psu_counts(
            rng,
            data.clusters_by_stratum,
            len(data.cluster_table),
            method,
        )
        total_weight = float(
            data.layer_cluster_weights["F_X"].dot(cluster_counts).sum()
        )
        participant_weight = float(
            data.layer_cluster_weights[
                "F_A_online_political_participants"
            ].dot(cluster_counts).sum()
        )
        participation_rows.append(
            {
                "replication": replication,
                "weighted_participation_rate": participant_weight / total_weight,
            }
        )
        for layer in LAYERS:
            score_weights = data.layer_cluster_weights[layer].dot(cluster_counts)
            rows.append(
                {
                    "replication": replication,
                    "layer": layer,
                    **summarize_sorted_distribution(
                        data.score_values, score_weights, delta_m
                    ),
                }
            )
        if progress_every > 0 and (
            replication % progress_every == 0 or replication == replications
        ):
            elapsed = time.perf_counter() - started
            print(
                "Bootstrap {0}/{1} ({2:.1f}%), elapsed {3:.1f}s".format(
                    replication,
                    replications,
                    100.0 * replication / replications,
                    elapsed,
                ),
                flush=True,
            )
    return pd.DataFrame(rows), pd.DataFrame(participation_rows)


def summarize_layers(
    point: pd.DataFrame,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for layer in LAYERS:
        point_row = point.loc[point["layer"].eq(layer)].iloc[0]
        layer_bootstrap = replicates.loc[replicates["layer"].eq(layer)]
        for metric in METRICS:
            values = pd.to_numeric(
                layer_bootstrap[metric], errors="coerce"
            ).dropna()
            rows.append(
                {
                    "layer": layer,
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
    upper = "F_A_online_political_participants"
    lower = "F_X"
    point_indexed = point.set_index("layer")
    indexed = replicates.set_index(["replication", "layer"])
    rows = []
    for metric in METRICS:
        estimate = float(
            point_indexed.loc[upper, metric] - point_indexed.loc[lower, metric]
        )
        upper_values = indexed.xs(upper, level="layer")[metric]
        lower_values = indexed.xs(lower, level="layer")[metric]
        differences = (upper_values - lower_values).dropna()
        probability_positive = float(np.mean(differences > 0))
        rows.append(
            {
                "contrast": "F_A_minus_F_X",
                "metric": metric,
                "estimate": estimate,
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


def summarize_participation(
    estimate: float,
    replicates: pd.DataFrame,
) -> pd.DataFrame:
    values = pd.to_numeric(
        replicates["weighted_participation_rate"], errors="coerce"
    ).dropna()
    return pd.DataFrame(
        [
            {
                "measure": "weighted_online_political_participation_rate",
                "estimate": estimate,
                "bootstrap_mean": values.mean(),
                "bootstrap_se": values.std(ddof=1),
                "ci95_low": values.quantile(0.025),
                "ci95_high": values.quantile(0.975),
                "bootstrap_replications_valid": len(values),
            }
        ]
    )


def distribution_source_data(data: PreparedData) -> pd.DataFrame:
    ones = np.ones(len(data.cluster_table), dtype=float)
    rows = []
    for layer in LAYERS:
        weights = data.layer_cluster_weights[layer].dot(ones)
        normalized = weights / weights.sum()
        rows.extend(
            {
                "layer": layer,
                "opinion": opinion,
                "probability_mass": probability,
            }
            for opinion, probability in zip(data.score_values, normalized)
            if probability > 0
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate ANES 2020 F_X -> F_A selection using general online "
            "political participation and a stratified PSU bootstrap."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--biden-col", default="V201151")
    parser.add_argument("--trump-col", default="V201152")
    parser.add_argument("--participation-col", default="V202029")
    parser.add_argument("--participant-value", type=float, default=1.0)
    parser.add_argument("--weight-col", default="V200010b")
    parser.add_argument("--psu-col", default="V200010c")
    parser.add_argument("--strata-col", default="V200010d")
    parser.add_argument("--delta-m", type=float, default=0.15)
    parser.add_argument("--bootstrap-replications", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--bootstrap-method",
        choices=["rao_wu", "ordinary"],
        default="rao_wu",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N replications; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replications < 1:
        raise ValueError("bootstrap-replications must be positive.")
    if not 0 < args.delta_m < 0.5:
        raise ValueError("delta-m must lie strictly between 0 and 0.5.")

    data, diagnostics = prepare_analysis_data(
        input_csv=args.input_csv,
        biden_col=args.biden_col,
        trump_col=args.trump_col,
        participation_col=args.participation_col,
        participant_value=args.participant_value,
        weight_col=args.weight_col,
        psu_col=args.psu_col,
        strata_col=args.strata_col,
    )
    point = point_estimates(data, args.delta_m)
    replicates, participation_replicates = run_bootstrap(
        data=data,
        replications=args.bootstrap_replications,
        seed=args.seed,
        delta_m=args.delta_m,
        method=args.bootstrap_method,
        progress_every=args.progress_every,
    )
    layer_summary = summarize_layers(point, replicates)
    differences = summarize_differences(point, replicates)
    participation_summary = summarize_participation(
        float(diagnostics.loc[0, "weighted_participation_rate"]),
        participation_replicates,
    )
    distribution_data = distribution_source_data(data)

    diagnostics["delta_m"] = args.delta_m
    diagnostics["bootstrap_method"] = args.bootstrap_method
    diagnostics["bootstrap_replications"] = args.bootstrap_replications
    diagnostics["bootstrap_seed"] = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "anes2020_selection_analysis_sample.csv": data.sample,
        "anes2020_selection_cluster_table.csv": data.cluster_table,
        "anes2020_selection_point_estimates.csv": point,
        "anes2020_selection_bootstrap_replicates.csv": replicates,
        "anes2020_selection_participation_bootstrap_replicates.csv": (
            participation_replicates
        ),
        "anes2020_selection_layer_summary.csv": layer_summary,
        "anes2020_selection_differences.csv": differences,
        "anes2020_selection_participation_summary.csv": participation_summary,
        "anes2020_selection_distribution_source_data.csv": distribution_data,
        "anes2020_selection_diagnostics.csv": diagnostics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            args.output_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    print("\nANES 2020 selection analysis complete.")
    print("Output directory: {0}".format(args.output_dir))
    print("\nParticipation:")
    print(participation_summary.to_string(index=False))
    print("\nPrimary polarization contrasts (F_A - F_X):")
    print(
        differences.loc[
            differences["metric"].isin(["P", "BPW", "U_mu", "BTM"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
