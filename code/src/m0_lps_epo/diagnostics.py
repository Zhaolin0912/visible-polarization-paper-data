from __future__ import annotations

import numpy as np


def vma_top_share(v, top_share=0.10):
    v = np.asarray(v, dtype=float)
    total = float(np.sum(v))
    if total <= 0:
        return np.nan
    top_n = max(1, int(np.ceil(top_share * len(v))))
    top_sum = float(np.sum(np.sort(v)[-top_n:]))
    return top_sum / (top_share * total)


def gini_exposure(v):
    v = np.asarray(v, dtype=float)
    total = float(np.sum(v))
    if total <= 0:
        return np.nan
    x = np.sort(v)
    n = len(x)
    weighted_rank_sum = float(np.sum(np.arange(1, n + 1) * x))
    return (2.0 * weighted_rank_sum) / (n * total) - (n + 1.0) / n


def feed_diagnostics(active, m, D, M_R, M_V, K, v_V=None):
    candidate_sizes = np.array([len(items) for items in D], dtype=int)
    visible_R = np.array([len(items) for items in M_R], dtype=int)
    visible_V = np.array([len(items) for items in M_V], dtype=int)
    out = {
        "ParticipationRate": float(np.mean(active)),
        "TotalPosts": int(np.sum(m)),
        "MeanCandidates": float(np.mean(candidate_sizes)),
        "MedianCandidates": float(np.median(candidate_sizes)),
        "FeedSaturation": float(np.mean(candidate_sizes > K)),
        "MeanVisibleItems_R": float(np.mean(visible_R)),
        "MeanVisibleItems_V": float(np.mean(visible_V)),
        "ValidObserverRate_R": float(np.mean(visible_R > 0)),
        "ValidObserverRate_V": float(np.mean(visible_V > 0)),
    }
    if v_V is not None:
        out["VMA_10pct"] = vma_top_share(v_V, top_share=0.10)
        out["Gini_V"] = gini_exposure(v_V)
        out["ValidRate_VMA_Gini"] = float(np.sum(v_V) > 0)
    return out
