from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np


def adjacency_from_sets(adjacency_sets: List[Set[int]]) -> List[np.ndarray]:
    return [np.array(sorted(neighbors), dtype=int) for neighbors in adjacency_sets]


def generate_ba_adjacency(N: int, m_BA: int, rng: np.random.Generator) -> List[np.ndarray]:
    if N <= 0:
        raise ValueError("N must be positive.")
    if m_BA < 1:
        raise ValueError("m_BA must be at least 1.")
    if m_BA >= N:
        raise ValueError("m_BA must be smaller than N.")

    m0 = m_BA + 1
    adjacency_sets: List[Set[int]] = [set() for _ in range(N)]

    for i in range(m0):
        for j in range(i + 1, m0):
            adjacency_sets[i].add(j)
            adjacency_sets[j].add(i)

    for new_node in range(m0, N):
        degrees = np.array([len(adjacency_sets[i]) for i in range(new_node)], dtype=float)
        probs = degrees / degrees.sum()
        targets = rng.choice(new_node, size=m_BA, replace=False, p=probs)
        for target in targets:
            target_i = int(target)
            adjacency_sets[new_node].add(target_i)
            adjacency_sets[target_i].add(new_node)

    return adjacency_from_sets(adjacency_sets)


def generate_homophilous_ba_adjacency(
    N: int,
    m_BA: int,
    rng: np.random.Generator,
    x: np.ndarray,
    homophily_chi: float = 4.0,
    attachment_epsilon: float = 1.0,
) -> List[np.ndarray]:
    if N <= 0:
        raise ValueError("N must be positive.")
    if m_BA < 1:
        raise ValueError("m_BA must be at least 1.")
    if m_BA >= N:
        raise ValueError("m_BA must be smaller than N.")
    if len(x) != N:
        raise ValueError("x must have length N.")
    if attachment_epsilon < 0.0:
        raise ValueError("attachment_epsilon must be non-negative.")

    m0 = m_BA + 1
    adjacency_sets: List[Set[int]] = [set() for _ in range(N)]

    for i in range(m0):
        for j in range(i + 1, m0):
            adjacency_sets[i].add(j)
            adjacency_sets[j].add(i)

    for new_node in range(m0, N):
        existing = np.arange(new_node, dtype=int)
        degrees = np.array([len(adjacency_sets[i]) for i in existing], dtype=float)
        degree_term = degrees + float(attachment_epsilon)
        similarity_term = np.exp(-float(homophily_chi) * np.abs(x[existing] - x[new_node]))
        weights = degree_term * similarity_term
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            probs = np.full(new_node, 1.0 / new_node, dtype=float)
        else:
            probs = weights / weight_sum
        targets = rng.choice(existing, size=m_BA, replace=False, p=probs)
        for target in targets:
            target_i = int(target)
            adjacency_sets[new_node].add(target_i)
            adjacency_sets[target_i].add(new_node)

    return adjacency_from_sets(adjacency_sets)


def generate_er_adjacency(N: int, m_BA: int, rng: np.random.Generator, p_ER: Optional[float] = None) -> List[np.ndarray]:
    if N <= 0:
        raise ValueError("N must be positive.")
    p = (2.0 * m_BA) / (N - 1) if p_ER is None else float(p_ER)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_ER must be in [0, 1].")

    adjacency_sets: List[Set[int]] = [set() for _ in range(N)]
    if p == 0.0:
        return adjacency_from_sets(adjacency_sets)
    if p == 1.0:
        for i in range(N):
            adjacency_sets[i].update(j for j in range(N) if j != i)
        return adjacency_from_sets(adjacency_sets)

    # Fast G(n,p) edge skipping; expected mean degree matches BA at about 2*m_BA.
    log_q = np.log1p(-p)
    v = 1
    w = -1
    while v < N:
        w += 1 + int(np.floor(np.log1p(-rng.random()) / log_q))
        while w >= v and v < N:
            w -= v
            v += 1
        if v < N:
            adjacency_sets[v].add(w)
            adjacency_sets[w].add(v)

    return adjacency_from_sets(adjacency_sets)


def generate_ws_adjacency(
    N: int,
    m_BA: int,
    rng: np.random.Generator,
    k_WS: Optional[int] = None,
    p_WS: float = 0.1,
) -> List[np.ndarray]:
    if N <= 0:
        raise ValueError("N must be positive.")
    if not 0.0 <= p_WS <= 1.0:
        raise ValueError("p_WS must be in [0, 1].")

    k = 2 * m_BA if k_WS is None else int(k_WS)
    k = min(k, N - 1)
    if k % 2 == 1:
        k -= 1
    if k < 2:
        raise ValueError("k_WS must be at least 2 after making it even.")

    adjacency_sets: List[Set[int]] = [set() for _ in range(N)]
    ring_edges: List[Tuple[int, int]] = []
    for i in range(N):
        for offset in range(1, k // 2 + 1):
            j = (i + offset) % N
            adjacency_sets[i].add(j)
            adjacency_sets[j].add(i)
            ring_edges.append((i, j))

    # Rewire only the clockwise ring edges, preserving edge count while creating shortcuts.
    for i, j in ring_edges:
        if rng.random() >= p_WS:
            continue
        if j not in adjacency_sets[i]:
            continue

        excluded = adjacency_sets[i] | {i}
        available = np.array([node for node in range(N) if node not in excluded], dtype=int)
        if len(available) == 0:
            continue

        new_j = int(rng.choice(available))
        adjacency_sets[i].remove(j)
        adjacency_sets[j].remove(i)
        adjacency_sets[i].add(new_j)
        adjacency_sets[new_j].add(i)

    return adjacency_from_sets(adjacency_sets)


def generate_network_adjacency(
    network_type: str,
    N: int,
    m_BA: int,
    rng: np.random.Generator,
    x: Optional[np.ndarray] = None,
    p_ER: Optional[float] = None,
    k_WS: Optional[int] = None,
    p_WS: float = 0.1,
    homophily_chi: float = 0.0,
    attachment_epsilon: float = 1.0,
) -> List[np.ndarray]:
    network = network_type.upper()
    if network == "BA":
        return generate_ba_adjacency(N=N, m_BA=m_BA, rng=rng)
    if network in {"BA_HOMOPHILY", "HBA", "HOMO_BA"}:
        if x is None:
            raise ValueError("Homophilous BA network requires private opinions x.")
        return generate_homophilous_ba_adjacency(
            N=N,
            m_BA=m_BA,
            rng=rng,
            x=x,
            homophily_chi=homophily_chi,
            attachment_epsilon=attachment_epsilon,
        )
    if network == "ER":
        return generate_er_adjacency(N=N, m_BA=m_BA, rng=rng, p_ER=p_ER)
    if network == "WS":
        return generate_ws_adjacency(N=N, m_BA=m_BA, rng=rng, k_WS=k_WS, p_WS=p_WS)
    raise NotImplementedError(f"Unknown network type: {network_type}")


def degree_array(adjacency: List[np.ndarray]) -> np.ndarray:
    return np.array([len(neighbors) for neighbors in adjacency], dtype=float)


def standardized_log_degree(k: np.ndarray) -> np.ndarray:
    logk = np.log1p(k)
    sd = float(logk.std())
    if sd == 0.0:
        return np.zeros_like(logk)
    return (logk - logk.mean()) / sd
