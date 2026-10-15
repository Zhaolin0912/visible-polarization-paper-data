from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Node visibility diagnostics require networkx. "
            "Install it in your environment with: pip install networkx"
        ) from exc
    return nx

#把邻接表转成 NetworkX 图
def adjacency_to_graph(adjacency: List[np.ndarray]):
    nx = require_networkx()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(adjacency)))
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            j = int(j)
            if i < j:
                graph.add_edge(i, j)
    return graph


def _safe_eigenvector_centrality(nx, graph):
    try:
        return nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
    except Exception:
        return {node: np.nan for node in graph.nodes}


def _detect_communities(nx, graph, seed=12345, method="louvain"):
    method = str(method).lower()
    if method == "louvain" and hasattr(nx.community, "louvain_communities"):
        return list(nx.community.louvain_communities(graph, seed=seed))
    if method in {"label_propagation", "lpa", "asyn_lpa"}:
        return list(nx.community.asyn_lpa_communities(graph, seed=seed))
    try:
        return list(nx.community.greedy_modularity_communities(graph))
    except Exception:
        return [set(graph.nodes)]


def compute_network_node_metrics(
    adjacency: List[np.ndarray],
    betweenness_k=200,
    seed=12345,
    community_method="louvain",
):
    nx = require_networkx()
    graph = adjacency_to_graph(adjacency)
    N = graph.number_of_nodes()
    degree = dict(graph.degree())
    log_degree = {node: float(np.log1p(deg)) for node, deg in degree.items()}

    if betweenness_k is None or int(betweenness_k) >= N:
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        betweenness_sample_k = N
    else:
        betweenness_sample_k = max(1, int(betweenness_k))
        betweenness = nx.betweenness_centrality(
            graph,
            k=betweenness_sample_k,
            normalized=True,
            seed=seed,
        )

    closeness = nx.closeness_centrality(graph)
    pagerank = nx.pagerank(graph)
    eigenvector = _safe_eigenvector_centrality(nx, graph)
    core_number = nx.core_number(graph)
    communities = _detect_communities(nx, graph, seed=seed, method=community_method)

    community_id = {}
    community_size = {}
    for cid, members in enumerate(communities):
        size = len(members)
        for node in members:
            community_id[int(node)] = cid
            community_size[int(node)] = size

    rows = []
    for node in range(N):
        rows.append(
            {
                "node_id": node,
                "network_degree": float(degree.get(node, 0.0)),
                "network_log_degree": float(log_degree.get(node, 0.0)),
                "betweenness_centrality": float(betweenness.get(node, np.nan)),
                "betweenness_sample_k": betweenness_sample_k,
                "closeness_centrality": float(closeness.get(node, np.nan)),
                "pagerank_centrality": float(pagerank.get(node, np.nan)),
                "eigenvector_centrality": float(eigenvector.get(node, np.nan)),
                "core_number": float(core_number.get(node, np.nan)),
                "community_id": int(community_id.get(node, -1)),
                "community_size": int(community_size.get(node, 1)),
            }
        )
    return pd.DataFrame(rows)


def node_visibility_correlations(
    node_df: pd.DataFrame,
    exposure_columns: Iterable[str] = ("mean_v_R", "mean_v_V", "mean_algorithm_lift"),
    attribute_columns: Iterable[str] = (
        "degree",
        "network_log_degree",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank_centrality",
        "eigenvector_centrality",
        "core_number",
        "community_size",
        "x_extremity",
        "mean_y_extremity",
        "mean_posts",
    ),
):
    rows = []
    for exposure in exposure_columns:
        if exposure not in node_df.columns:
            continue
        for attribute in attribute_columns:
            if attribute not in node_df.columns:
                continue
            pair = node_df[[exposure, attribute]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair[exposure].nunique() < 2 or pair[attribute].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = float(pair[exposure].corr(pair[attribute], method="pearson"))
                spearman = float(pair[exposure].corr(pair[attribute], method="spearman"))
            rows.append(
                {
                    "exposure": exposure,
                    "attribute": attribute,
                    "n": int(len(pair)),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def community_visibility_summary(node_df: pd.DataFrame):
    total_nodes = len(node_df)
    node_df = node_df.copy()
    node_df["positive_algorithm_lift"] = node_df["total_algorithm_lift"].clip(lower=0.0)
    node_df["negative_algorithm_lift"] = (-node_df["total_algorithm_lift"].clip(upper=0.0))
    node_df["absolute_algorithm_lift"] = node_df["total_algorithm_lift"].abs()
    totals = {
        "total_v_R": float(node_df["total_v_R"].sum()),
        "total_v_V": float(node_df["total_v_V"].sum()),
        "total_algorithm_lift": float(node_df["total_algorithm_lift"].sum()),
        "positive_algorithm_lift": float(node_df["positive_algorithm_lift"].sum()),
        "negative_algorithm_lift": float(node_df["negative_algorithm_lift"].sum()),
        "absolute_algorithm_lift": float(node_df["absolute_algorithm_lift"].sum()),
        "total_posts": float(node_df["total_posts"].sum()),
    }
    grouped = node_df.groupby("community_id", as_index=False)
    out = grouped.agg(
        community_size=("node_id", "size"),
        mean_degree=("degree", "mean"),
        mean_betweenness=("betweenness_centrality", "mean"),
        mean_closeness=("closeness_centrality", "mean"),
        mean_pagerank=("pagerank_centrality", "mean"),
        mean_core_number=("core_number", "mean"),
        mean_x=("x", "mean"),
        mean_x_extremity=("x_extremity", "mean"),
        mean_y=("mean_y", "mean"),
        mean_y_extremity=("mean_y_extremity", "mean"),
        total_posts=("total_posts", "sum"),
        total_v_R=("total_v_R", "sum"),
        total_v_V=("total_v_V", "sum"),
        total_algorithm_lift=("total_algorithm_lift", "sum"),
        positive_algorithm_lift=("positive_algorithm_lift", "sum"),
        negative_algorithm_lift=("negative_algorithm_lift", "sum"),
        absolute_algorithm_lift=("absolute_algorithm_lift", "sum"),
        mean_v_R=("mean_v_R", "mean"),
        mean_v_V=("mean_v_V", "mean"),
        mean_algorithm_lift=("mean_algorithm_lift", "mean"),
    )
    out["node_share"] = out["community_size"] / total_nodes
    out["post_share"] = np.nan if totals["total_posts"] == 0 else out["total_posts"] / totals["total_posts"]
    out["exposure_share_R"] = np.nan if totals["total_v_R"] == 0 else out["total_v_R"] / totals["total_v_R"]
    out["exposure_share_V"] = np.nan if totals["total_v_V"] == 0 else out["total_v_V"] / totals["total_v_V"]
    out["positive_algorithm_lift_share"] = (
        np.nan
        if totals["positive_algorithm_lift"] == 0
        else out["positive_algorithm_lift"] / totals["positive_algorithm_lift"]
    )
    out["negative_algorithm_lift_share"] = (
        np.nan
        if totals["negative_algorithm_lift"] == 0
        else out["negative_algorithm_lift"] / totals["negative_algorithm_lift"]
    )
    out["absolute_algorithm_lift_share"] = (
        np.nan
        if totals["absolute_algorithm_lift"] == 0
        else out["absolute_algorithm_lift"] / totals["absolute_algorithm_lift"]
    )
    out["overexposure_R"] = out["exposure_share_R"] - out["node_share"]
    out["overexposure_V"] = out["exposure_share_V"] - out["node_share"]
    out["positive_lift_over_node_share"] = out["positive_algorithm_lift_share"] - out["node_share"]
    out["negative_lift_over_node_share"] = out["negative_algorithm_lift_share"] - out["node_share"]
    out["absolute_lift_over_node_share"] = out["absolute_algorithm_lift_share"] - out["node_share"]
    return out


def community_visibility_correlations(community_df: pd.DataFrame):
    return node_visibility_correlations(
        community_df,
        exposure_columns=(
            "exposure_share_R",
            "exposure_share_V",
            "positive_algorithm_lift_share",
            "negative_algorithm_lift_share",
            "absolute_algorithm_lift_share",
            "mean_algorithm_lift",
        ),
        attribute_columns=(
            "community_size",
            "node_share",
            "mean_degree",
            "mean_betweenness",
            "mean_closeness",
            "mean_pagerank",
            "mean_core_number",
            "mean_x_extremity",
            "mean_y_extremity",
            "post_share",
        ),
    )
