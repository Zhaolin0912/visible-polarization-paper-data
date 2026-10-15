from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TWITTER_DIR = REPO_ROOT / "data" / "raw" / "twitter2020"
DEFAULT_ANES_FX = REPO_ROOT / "outputs" / "empirical" / "anes2020" / "anes2020_fx.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "empirical" / "twitter2020"

TWITTER_USECOLS = [
    "created_at",
    "tweet_id",
    "tweet",
    "likes",
    "retweet_count",
    "source",
    "user_id",
    "user_screen_name",
    "user_followers_count",
    "country",
    "state",
    "collected_at",
]

COMMON_ENGLISH_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "you",
    "are",
    "not",
    "have",
    "vote",
    "voted",
    "voting",
    "election",
    "president",
    "america",
    "american",
    "people",
}


def weighted_pairwise_distance(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    total_weight = w.sum()
    if len(x) == 0 or total_weight <= 0:
        return np.nan

    cum_w = np.cumsum(w)
    cum_wx = np.cumsum(w * x)
    total = 0.0
    for i in range(1, len(x)):
        total += w[i] * (x[i] * cum_w[i - 1] - cum_wx[i - 1])
    return float(2.0 * total / (total_weight**2))


def weighted_bpw(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    x = values[order]
    w = weights[order] / weights.sum()
    cum_prev = np.concatenate(([0.0], np.cumsum(w)[:-1]))
    cum_next = np.cumsum(w)
    left_mass = np.minimum(cum_next, 0.5) - np.minimum(cum_prev, 0.5)
    left_mass = np.maximum(left_mass, 0.0)
    right_mass = w - left_mass
    w1 = np.sum(left_mass * np.abs(x) + right_mass * np.abs(x - 1.0))
    return float(1.0 - 2.0 * w1)


def summarize_distribution(values: np.ndarray, weights: np.ndarray, delta_m: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return {
            "n": 0.0,
            "weight_sum": 0.0,
            "mu": np.nan,
            "P": np.nan,
            "BPW": np.nan,
            "U_mu": np.nan,
            "CON": np.nan,
            "moderate_share": np.nan,
            "L_deltaB": np.nan,
            "R_deltaB": np.nan,
            "S_deltaB": np.nan,
            "BAL_deltaB": np.nan,
            "BTM": np.nan,
        }

    weights = weights / weights.sum()
    mu = float(np.sum(values * weights))
    pairwise = weighted_pairwise_distance(values, weights)
    left_tail = float(np.sum(weights * (values < 0.5 - delta_m)))
    right_tail = float(np.sum(weights * (values > 0.5 + delta_m)))
    side_mass = left_tail + right_tail
    balance = 0.0 if side_mass <= 0 else float(2.0 * min(left_tail, right_tail) / side_mass)
    return {
        "n": float(len(values)),
        "weight_sum": float(weights.sum()),
        "mu": mu,
        "P": pairwise,
        "BPW": weighted_bpw(values, weights),
        "U_mu": float(2.0 * abs(mu - 0.5)),
        "CON": float(1.0 - 2.0 * pairwise),
        "moderate_share": float(np.sum(weights * (np.abs(values - 0.5) < delta_m))),
        "L_deltaB": left_tail,
        "R_deltaB": right_tail,
        "S_deltaB": side_mass,
        "BAL_deltaB": balance,
        "BTM": float(2.0 * min(left_tail, right_tail)),
    }


def is_probably_english(text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    cleaned = re.sub(r"https?://\S+|@\w+|#\w+", " ", text)
    letters = re.findall(r"[A-Za-z]", cleaned)
    if len(letters) < 20:
        return False
    ascii_chars = sum(1 for ch in cleaned if ord(ch) < 128)
    ascii_ratio = ascii_chars / max(len(cleaned), 1)
    words = re.findall(r"[A-Za-z]+", cleaned.lower())
    common_hits = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    return ascii_ratio >= 0.85 and common_hits >= 1


def make_tweet_key(row: pd.Series) -> str:
    """Create a stable text key because Kaggle tweet_id values are stored in scientific notation."""
    parts = [
        str(row.get("tweet_id", "")),
        str(row.get("created_at_dt", "")),
        str(row.get("user_screen_name", "")),
        str(row.get("tweet", "")),
    ]
    digest = hashlib.sha1("\n".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return f"tw_{digest[:20]}"


def load_hashtag_file(path: Path, source_label: str, chunksize: int) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=TWITTER_USECOLS,
        chunksize=chunksize,
        dtype={"tweet_id": "string", "user_id": "string"},
        engine="python",
        on_bad_lines="skip",
    ):
        chunk["hashtag_source"] = source_label
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=TWITTER_USECOLS + ["hashtag_source"])
    return pd.concat(chunks, ignore_index=True)


def prepare_twitter_layers(
    twitter_dir: Path,
    anes_fx_path: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    english_filter: str,
    both_hashtag_value: float,
    chunksize: int,
    delta_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trump = load_hashtag_file(twitter_dir / "hashtag_donaldtrump.csv", "trump", chunksize)
    biden = load_hashtag_file(twitter_dir / "hashtag_joebiden.csv", "biden", chunksize)
    raw = pd.concat([trump, biden], ignore_index=True)

    raw["created_at_dt"] = pd.to_datetime(raw["created_at"], errors="coerce")
    raw["likes"] = pd.to_numeric(raw["likes"], errors="coerce").fillna(0).clip(lower=0)
    raw["retweet_count"] = pd.to_numeric(raw["retweet_count"], errors="coerce").fillna(0).clip(lower=0)
    raw["user_followers_count"] = (
        pd.to_numeric(raw["user_followers_count"], errors="coerce").fillna(0).clip(lower=0)
    )
    raw = raw.loc[raw["created_at_dt"].notna() & raw["tweet_id"].notna() & raw["tweet"].notna()].copy()
    raw = raw.loc[
        (raw["created_at_dt"] >= pd.Timestamp(start_date))
        & (raw["created_at_dt"] <= pd.Timestamp(end_date))
    ].copy()

    # Merge duplicate records across the Biden and Trump hashtag files before assigning a stance proxy.
    group_cols = ["tweet_id"]
    agg = {
        "created_at_dt": "min",
        "tweet": "first",
        "likes": "max",
        "retweet_count": "max",
        "source": "first",
        "user_id": "first",
        "user_screen_name": "first",
        "user_followers_count": "max",
        "country": "first",
        "state": "first",
        "hashtag_source": lambda s: ",".join(sorted(set(s.dropna().astype(str)))),
    }
    tweets = raw.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    tweets["tweet_id_raw"] = tweets["tweet_id"].astype(str)
    tweets["tweet_key"] = tweets.apply(make_tweet_key, axis=1)
    tweets["has_biden_hashtag"] = tweets["hashtag_source"].str.contains("biden", regex=False)
    tweets["has_trump_hashtag"] = tweets["hashtag_source"].str.contains("trump", regex=False)
    tweets["hashtag_type"] = np.select(
        [
            tweets["has_biden_hashtag"] & tweets["has_trump_hashtag"],
            tweets["has_biden_hashtag"],
            tweets["has_trump_hashtag"],
        ],
        ["both", "biden_only", "trump_only"],
        default="unknown",
    )
    tweets["stance_proxy"] = np.select(
        [
            tweets["hashtag_type"].eq("biden_only"),
            tweets["hashtag_type"].eq("trump_only"),
            tweets["hashtag_type"].eq("both"),
        ],
        [0.0, 1.0, both_hashtag_value],
        default=np.nan,
    )

    if english_filter == "heuristic":
        tweets["is_probably_english"] = tweets["tweet"].map(is_probably_english)
        tweets = tweets.loc[tweets["is_probably_english"]].copy()
    else:
        tweets["is_probably_english"] = np.nan

    tweets["post_weight"] = 1.0
    tweets["engagement_weight"] = 1.0 + tweets["likes"] + tweets["retweet_count"]
    tweets["retweet_weight"] = 1.0 + tweets["retweet_count"]
    tweets["like_weight"] = 1.0 + tweets["likes"]

    # Author-uniform layer: each observed author contributes one mean public stance.
    author = (
        tweets.dropna(subset=["user_id", "stance_proxy"])
        .groupby("user_id", as_index=False)
        .agg(stance_proxy=("stance_proxy", "mean"), posts=("tweet_id", "count"))
    )
    author["author_weight"] = 1.0

    fx = pd.read_csv(anes_fx_path)
    fx_metrics = summarize_distribution(
        fx["x"].to_numpy(dtype=float), fx["weight"].to_numpy(dtype=float), delta_m
    )

    layers = {
        "F_Y_author_uniform": (
            author["stance_proxy"].to_numpy(dtype=float),
            author["author_weight"].to_numpy(dtype=float),
        ),
        "F_P_post_uniform": (
            tweets["stance_proxy"].to_numpy(dtype=float),
            tweets["post_weight"].to_numpy(dtype=float),
        ),
        "F_V_engagement_proxy": (
            tweets["stance_proxy"].to_numpy(dtype=float),
            tweets["engagement_weight"].to_numpy(dtype=float),
        ),
        "F_V_retweet_proxy": (
            tweets["stance_proxy"].to_numpy(dtype=float),
            tweets["retweet_weight"].to_numpy(dtype=float),
        ),
        "F_V_like_proxy": (
            tweets["stance_proxy"].to_numpy(dtype=float),
            tweets["like_weight"].to_numpy(dtype=float),
        ),
    }

    rows = []
    for layer_name, (values, weights) in layers.items():
        metrics = summarize_distribution(values, weights, delta_m)
        row = {
            "layer": layer_name,
            **metrics,
            "VRB_P_vs_ANES_FX": metrics["P"] - fx_metrics["P"],
            "VRB_BPW_vs_ANES_FX": metrics["BPW"] - fx_metrics["BPW"],
            "AMB_vs_ANES_FX": abs(metrics["mu"] - fx_metrics["mu"]),
            "MDI_vs_ANES_FX": fx_metrics["moderate_share"] - metrics["moderate_share"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(
        [
            {
                "raw_rows_after_parser": float(len(raw)),
                "unique_tweets_after_dedup": float(len(tweets)),
                "unique_authors": float(author["user_id"].nunique()),
                "biden_only_tweets": float(tweets["hashtag_type"].eq("biden_only").sum()),
                "trump_only_tweets": float(tweets["hashtag_type"].eq("trump_only").sum()),
                "both_hashtag_tweets": float(tweets["hashtag_type"].eq("both").sum()),
                "english_filter": english_filter,
                "both_hashtag_value": both_hashtag_value,
                "date_start": start_date,
                "date_end": end_date,
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tweets.to_csv(output_dir / "twitter2020_clean_tweets.csv", index=False, encoding="utf-8-sig")
    author.to_csv(output_dir / "twitter2020_author_stance_proxy.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "twitter2020_layer_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "twitter2020_cleaning_diagnostics.csv", index=False, encoding="utf-8-sig")
    return summary, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean Kaggle US Election 2020 Tweets and construct empirical public-expression "
            "and visibility-proxy layers."
        )
    )
    parser.add_argument("--twitter-dir", type=Path, default=DEFAULT_TWITTER_DIR)
    parser.add_argument("--anes-fx", type=Path, default=DEFAULT_ANES_FX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2020-10-15 00:00:00")
    parser.add_argument("--end-date", default="2020-11-08 23:59:59")
    parser.add_argument("--english-filter", choices=["none", "heuristic"], default="heuristic")
    parser.add_argument("--both-hashtag-value", type=float, default=0.5)
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--delta-m", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, diagnostics = prepare_twitter_layers(
        twitter_dir=args.twitter_dir,
        anes_fx_path=args.anes_fx,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        english_filter=args.english_filter,
        both_hashtag_value=args.both_hashtag_value,
        chunksize=args.chunksize,
        delta_m=args.delta_m,
    )
    print(f"Output directory: {args.output_dir}")
    print("\nCleaning diagnostics:")
    print(diagnostics.T.to_string(header=False))
    print("\nLayer summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
