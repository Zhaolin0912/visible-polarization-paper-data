from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_twitter2020_layers import summarize_distribution


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TWEETS = REPO_ROOT / "outputs" / "empirical" / "twitter2020" / "twitter2020_clean_tweets.csv"
DEFAULT_LABELS = (
    REPO_ROOT
    / "outputs"
    / "empirical"
    / "twitter2020"
    / "llm"
    / "deepseek_labels_tweetkey.jsonl"
)
DEFAULT_ANES_FX = REPO_ROOT / "outputs" / "empirical" / "anes2020" / "anes2020_fx.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "empirical" / "twitter2020" / "llm_tweetkey"


def add_author_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable author identifier without relying on rounded numeric IDs."""
    out = df.copy()
    screen_name = out.get("user_screen_name", pd.Series("", index=out.index, dtype="string"))
    screen_name = screen_name.astype("string").fillna("").str.strip().str.lower()
    user_id = out.get("user_id", pd.Series("", index=out.index, dtype="string"))
    user_id = user_id.astype("string").fillna("").str.strip().str.lower()
    tweet_key = out["tweet_key"].astype("string").fillna("")

    valid_screen_name = ~screen_name.isin(["", "nan", "none", "<na>"])
    valid_user_id = ~user_id.isin(["", "nan", "none", "<na>"])
    out["author_key_source"] = np.select(
        [valid_screen_name, valid_user_id],
        ["screen_name", "user_id"],
        default="tweet_key",
    )
    out["author_key"] = np.where(
        valid_screen_name,
        "sn:" + screen_name,
        np.where(valid_user_id, "id:" + user_id, "tw:" + tweet_key),
    )
    return out


def read_labels_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame(
            columns=["tweet_key", "tweet_id", "stance_score", "stance_label", "confidence", "usable"]
        )
    labels = pd.DataFrame(rows)
    if "tweet_key" not in labels.columns:
        labels["tweet_key"] = labels.get("item_id", labels.get("tweet_id")).astype(str)
    labels["tweet_id"] = labels["tweet_id"].astype(str)
    labels["tweet_key"] = labels["tweet_key"].astype(str)
    labels["stance_score"] = pd.to_numeric(labels["stance_score"], errors="coerce")
    labels["confidence"] = pd.to_numeric(labels["confidence"], errors="coerce")
    if "usable" not in labels.columns:
        labels["usable"] = True
    labels["usable"] = labels["usable"].fillna(False).astype(bool)
    labels = labels.drop_duplicates("tweet_key", keep="last")
    return labels


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["likes"] = pd.to_numeric(out["likes"], errors="coerce").fillna(0).clip(lower=0)
    out["retweet_count"] = (
        pd.to_numeric(out["retweet_count"], errors="coerce").fillna(0).clip(lower=0)
    )
    out["post_weight"] = 1.0
    out["engagement_weight"] = 1.0 + out["likes"] + out["retweet_count"]
    out["retweet_weight"] = 1.0 + out["retweet_count"]
    out["like_weight"] = 1.0 + out["likes"]
    return out


def add_author_uniform_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Give every author unit total mass while retaining tweet-level stances."""
    out = df.copy()
    posts_per_author = out.groupby("author_key")["tweet_key"].transform("count")
    if posts_per_author.le(0).any():
        raise ValueError("Every analyzed author must have at least one tweet.")
    out["author_uniform_weight"] = 1.0 / posts_per_author.astype(float)
    return out


def merge_tweets_and_labels(tweets: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["tweet_key", "tweet_id", "stance_score", "stance_label", "confidence", "usable"]
    merged_by_key = tweets.merge(
        labels[label_cols].drop(columns=["tweet_id"]),
        on="tweet_key",
        how="inner",
    )
    matched_keys = set(merged_by_key["tweet_key"].astype(str))

    # Older label files used tweet_id before tweet_key existed; keep them readable.
    old_labels = labels.loc[~labels["tweet_key"].astype(str).isin(matched_keys)].copy()
    old_labels = old_labels.drop(columns=["tweet_key"])
    merged_by_id = tweets.merge(
        old_labels,
        on="tweet_id",
        how="inner",
    )
    merged = pd.concat([merged_by_key, merged_by_id], ignore_index=True, sort=False)
    return merged.drop_duplicates("tweet_key", keep="last")


def build_summary(
    labeled: pd.DataFrame,
    anes_fx: pd.DataFrame,
    delta_m: float,
) -> pd.DataFrame:
    fx_metrics = summarize_distribution(
        anes_fx["x"].to_numpy(dtype=float), anes_fx["weight"].to_numpy(dtype=float), delta_m
    )

    if "author_uniform_weight" not in labeled.columns:
        labeled = add_author_uniform_weights(labeled)

    layers = {
        "F_Y_author_uniform": (
            labeled["stance_score"].to_numpy(dtype=float),
            labeled["author_uniform_weight"].to_numpy(dtype=float),
        ),
        "F_P_post_uniform": (
            labeled["stance_score"].to_numpy(dtype=float),
            labeled["post_weight"].to_numpy(dtype=float),
        ),
        "F_V_engagement_proxy": (
            labeled["stance_score"].to_numpy(dtype=float),
            labeled["engagement_weight"].to_numpy(dtype=float),
        ),
        "F_V_retweet_proxy": (
            labeled["stance_score"].to_numpy(dtype=float),
            labeled["retweet_weight"].to_numpy(dtype=float),
        ),
        "F_V_like_proxy": (
            labeled["stance_score"].to_numpy(dtype=float),
            labeled["like_weight"].to_numpy(dtype=float),
        ),
    }

    rows = []
    for layer_name, (values, weights) in layers.items():
        metrics = summarize_distribution(values, weights, delta_m)
        rows.append(
            {
                "layer": layer_name,
                **metrics,
                "VRB_P_vs_ANES_FX": metrics["P"] - fx_metrics["P"],
                "VRB_BPW_vs_ANES_FX": metrics["BPW"] - fx_metrics["BPW"],
                "AMB_vs_ANES_FX": abs(metrics["mu"] - fx_metrics["mu"]),
                "MDI_vs_ANES_FX": fx_metrics["moderate_share"] - metrics["moderate_share"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge DeepSeek stance labels back into Twitter 2020 layers and summarize metrics."
    )
    parser.add_argument("--tweets", type=Path, default=DEFAULT_TWEETS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--anes-fx", type=Path, default=DEFAULT_ANES_FX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--delta-m", type=float, default=0.15)
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help="Include labels marked usable=false. Default excludes them.",
    )
    args = parser.parse_args()

    tweet_cols = [
        "tweet_id",
        "tweet_key",
        "tweet_id_raw",
        "created_at_dt",
        "tweet",
        "hashtag_type",
        "likes",
        "retweet_count",
        "user_id",
        "user_screen_name",
    ]
    available_cols = pd.read_csv(args.tweets, nrows=0).columns
    usecols = [col for col in tweet_cols if col in available_cols]
    tweets = pd.read_csv(args.tweets, usecols=usecols, dtype={"tweet_id": "string", "user_id": "string"})
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)
    if "tweet_key" not in tweets.columns:
        tweets["tweet_key"] = tweets["tweet_id"].astype(str)
    tweets["tweet_key"] = tweets["tweet_key"].astype(str)
    labels = read_labels_jsonl(args.labels)
    merged = merge_tweets_and_labels(tweets, labels)
    merged = add_weights(merged)
    merged = add_author_key(merged)

    mask = merged["confidence"].ge(args.min_confidence) & merged["stance_score"].between(0, 1)
    if not args.include_unusable:
        mask &= merged["usable"]
    analysis = merged.loc[mask].copy()
    analysis = add_author_uniform_weights(analysis)

    anes_fx = pd.read_csv(args.anes_fx)
    summary = build_summary(analysis, anes_fx, delta_m=args.delta_m)
    diagnostics = pd.DataFrame(
        [
            {
                "tweets_total_clean": float(len(tweets)),
                "labels_total": float(len(labels)),
                "merged_labeled_rows": float(len(merged)),
                "analysis_rows": float(len(analysis)),
                "unique_authors_in_analysis": float(analysis["author_key"].nunique()),
                "F_Y_author_weight_sum": float(analysis["author_uniform_weight"].sum()),
                "min_confidence": args.min_confidence,
                "include_unusable": bool(args.include_unusable),
                "usable_rate_in_merged": float(merged["usable"].mean()) if len(merged) else np.nan,
                "mean_confidence_in_analysis": float(analysis["confidence"].mean()) if len(analysis) else np.nan,
                "author_key_screen_name_rate": float(
                    analysis["author_key_source"].eq("screen_name").mean()
                )
                if len(analysis)
                else np.nan,
            }
        ]
    )
    label_counts = (
        analysis["stance_label"]
        .value_counts(dropna=False)
        .rename_axis("stance_label")
        .reset_index(name="n")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(args.output_dir / "twitter2020_llm_labeled_tweets_analysis.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "twitter2020_llm_layer_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(args.output_dir / "twitter2020_llm_diagnostics.csv", index=False, encoding="utf-8-sig")
    label_counts.to_csv(args.output_dir / "twitter2020_llm_label_counts.csv", index=False, encoding="utf-8-sig")

    print(f"Output directory: {args.output_dir}")
    print("\nDiagnostics:")
    print(diagnostics.T.to_string(header=False))
    print("\nLabel counts:")
    print(label_counts.to_string(index=False))
    print("\nLayer summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
