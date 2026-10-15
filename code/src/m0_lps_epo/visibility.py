#模拟平台如何从邻居发出的候选内容中，为每个用户选择实际可见的内容；
#它同时生成随机曝光基准和算法曝光机制，用来比较算法是否放大某些观点或用户
from __future__ import annotations

from typing import List, Tuple

import numpy as np

#这是一个类型别名。它表示一个“可见内容项”由两个整数构成：(producer, item_id)
VisibleItem = Tuple[int, int]

#构建每个用户的信息流候选池
def build_candidate_items(adjacency: List[np.ndarray], m: np.ndarray) -> Tuple[List[List[VisibleItem]], np.ndarray]:
    D: List[List[VisibleItem]] = []
    sizes = np.zeros(len(adjacency), dtype=int)

    for i, neighbors in enumerate(adjacency):
        items: List[VisibleItem] = []
        for j in neighbors:
            producer = int(j)
            for item_id in range(int(m[producer])):
                items.append((producer, item_id))
        D.append(items)
        sizes[i] = len(items)

    return D, sizes

#随机曝光M_R
def sample_random_visibility(D: List[List[VisibleItem]], K: int, rng: np.random.Generator) -> List[List[VisibleItem]]:
    out: List[List[VisibleItem]] = []
    for items in D:
        n = len(items)
        if n <= K:
            out.append(list(items))
            continue
        idx = rng.choice(n, size=K, replace=False)
        out.append([items[int(k)] for k in idx])
    return out

#算法曝光 M_V
def sample_algorithm_visibility(
    D: List[List[VisibleItem]],
    K: int,
    y: np.ndarray,
    pi: np.ndarray,
    beta_E: float,
    beta_P: float,
    beta_G: float,
    producer_popularity: np.ndarray,
    rng: np.random.Generator,
) -> List[List[VisibleItem]]:
    out: List[List[VisibleItem]] = []
    for observer, items in enumerate(D):
        n = len(items)
        if n <= K:
            out.append(list(items))
            continue

        log_weights = np.empty(n, dtype=float)
        for idx, (producer, _item_id) in enumerate(items):
            extremity = 2.0 * abs(y[producer] - 0.5)
            profile_match = 1.0 - abs(y[producer] - pi[observer])
            popularity = producer_popularity[producer]
            log_weights[idx] = beta_E * extremity + beta_P * profile_match + beta_G * popularity

        weights = np.exp(log_weights - np.max(log_weights)) #更稳定，防止溢出，但是概率不变
        probs = weights / weights.sum()
        chosen = rng.choice(n, size=K, replace=False, p=probs)
        out.append([items[int(k)] for k in chosen])
    return out

#统计每个用户被曝光多少次v_R / v_V
def exposure_counts(M: List[List[VisibleItem]], N: int) -> np.ndarray:
    counts = np.zeros(N, dtype=int)
    for items in M:
        for producer, _item_id in items:
            counts[producer] += 1
    return counts

#选择实际使用哪种曝光 M_actual
def select_actual_visibility(M_R, M_V, actual_visibility: str):
    if actual_visibility == "algorithm":
        return M_V
    if actual_visibility == "random":
        return M_R
    raise ValueError(f"Unknown actual_visibility: {actual_visibility}")
