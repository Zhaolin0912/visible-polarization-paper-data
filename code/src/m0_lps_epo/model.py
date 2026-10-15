from __future__ import annotations

import numpy as np

from .agents import clip01, draw_activity_tendency, draw_cost, draw_private_opinion, draw_profile
from .diagnostics import feed_diagnostics
from .metrics import (
    accounting_decomposition,
    decomposition_closure_error,
    distribution_differences,
    observer_prb_stats,
    weighted_pairwise_distance,
)
from .network import degree_array, generate_network_adjacency, standardized_log_degree
from .transitions import default_bin_edges, transition_records_for_visibility
from .visibility import (
    build_candidate_items,
    exposure_counts,
    sample_algorithm_visibility,
    sample_random_visibility,
    select_actual_visibility,
)

#把选择发言变成[0,1]概率
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -700, 700)))

#感知一致性H定义 影响用户是否愿意参与
def compute_H_item(x, y_prev, M_prev):
    N = len(x)
    H = np.full(N, 0.5, dtype=float)
    dmax = np.maximum(x, 1.0 - x)

    for i, items in enumerate(M_prev):
        if not items:
            continue
        congruence = [1.0 - abs(x[i] - y_prev[producer]) / dmax[i] for producer, _item_id in items]
        H[i] = float(np.mean(congruence))
    return np.clip(H, 0.0, 1.0)


# Source-deduplicated H gives each visible producer one vote, regardless of repeated items.
def compute_H_source(x, y_prev, M_prev):
    N = len(x)
    H = np.full(N, 0.5, dtype=float)
    dmax = np.maximum(x, 1.0 - x)

    for i, items in enumerate(M_prev):
        if not items:
            continue
        producers = list(dict.fromkeys(producer for producer, _item_id in items))
        congruence = [1.0 - abs(x[i] - y_prev[producer]) / dmax[i] for producer in producers]
        H[i] = float(np.mean(congruence))
    return np.clip(H, 0.0, 1.0)


# Side-based H retains item weighting but replaces continuous distance with same-side agreement.
def compute_H_side(x, y_prev, M_prev):
    N = len(x)
    H = np.full(N, 0.5, dtype=float)

    for i, items in enumerate(M_prev):
        if not items:
            continue
        own_side = np.sign(x[i] - 0.5)
        same_side = [np.sign(y_prev[producer] - 0.5) == own_side for producer, _item_id in items]
        H[i] = float(np.mean(same_side))
    return np.clip(H, 0.0, 1.0)


def compute_perceived_congruence(x, y_prev, M_prev, definition="item"):
    definition = str(definition).strip().lower()
    if definition in {"item", "item_weighted", "item-weighted"}:
        return compute_H_item(x, y_prev, M_prev)
    if definition in {"source", "source_deduplicated", "source-deduplicated"}:
        return compute_H_source(x, y_prev, M_prev)
    if definition in {"side", "side_based", "side-based"}:
        return compute_H_side(x, y_prev, M_prev)
    raise ValueError(f"Unknown congruence_type: {definition}")

#计算用户看到内容的平均立场 影响用户下一轮如何表达
def compute_S_mean(y_prev, M_prev):
    S = np.full(len(y_prev), 0.5, dtype=float)
    for i, items in enumerate(M_prev):
        if not items:
            continue
        S[i] = float(np.mean([y_prev[producer] for producer, _item_id in items]))
    return S


def compute_perceived_minority_indicator(x, S_mean, definition):
    definition = str(definition).strip().lower()
    if definition in {"none", "", "disabled"}:
        return np.zeros(len(x), dtype=float)
    if definition in {"mean_side", "visible_mean_side"}:
        own_side = np.sign(x - 0.5)
        visible_side = np.sign(S_mean - 0.5)
        indicator = (own_side != 0.0) & (visible_side != 0.0) & (own_side != visible_side)
        return indicator.astype(float)
    raise ValueError(f"Unknown minority_definition: {definition}")


def compute_S_hard_dominant(y_prev, M_prev, tau_S=0.1, delta_S=0.0, n_dom_min=3):
    # 按内容项识别局部主导侧；中性内容保留在正负侧占比的共同分母中。
    tau_S = float(tau_S)
    delta_S = float(delta_S)
    n_dom_min = int(n_dom_min)
    if not 0.0 <= tau_S <= 1.0:
        raise ValueError("tau_S must be in [0, 1]")
    if not 0.0 <= delta_S < 0.5:
        raise ValueError("delta_S must be in [0, 0.5)")
    if n_dom_min < 1:
        raise ValueError("n_dom_min must be at least 1")

    S = compute_S_mean(y_prev, M_prev)
    dominant_used = np.zeros(len(y_prev), dtype=bool)
    upper = 0.5 + delta_S
    lower = 0.5 - delta_S

    for i, items in enumerate(M_prev):
        if not items:
            continue
        values = np.asarray([y_prev[producer] for producer, _item_id in items], dtype=float)
        positive = values[values > upper]
        negative = values[values < lower]
        advantage = (len(positive) - len(negative)) / len(values)

        if advantage > tau_S and len(positive) >= n_dom_min:
            S[i] = float(np.mean(positive))
            dominant_used[i] = True
        elif -advantage > tau_S and len(negative) >= n_dom_min:
            S[i] = float(np.mean(negative))
            dominant_used[i] = True

    return S, dominant_used


def compute_S_popular_cluster(x, y_prev, M_prev, kappa=2):
    # A popular-not-average target selects the locally densest visible opinion exemplar.
    # When several equally dense exemplars exist, private opinion x_i breaks the tie.
    kappa = int(kappa)
    if kappa < 1:
        raise ValueError("kappa must be at least 1")

    S = np.full(len(y_prev), 0.5, dtype=float)
    popular_used = np.zeros(len(y_prev), dtype=bool)

    for i, items in enumerate(M_prev):
        if not items:
            continue

        values = np.asarray([y_prev[producer] for producer, _item_id in items], dtype=float)
        n_values = len(values)

        if n_values == 1:
            S[i] = float(values[0])
            popular_used[i] = True
            continue

        if n_values == 2:
            # With only two visible opinions, fall back to the exemplar closer to x_i.
            distances_to_private = np.abs(values - x[i])
            min_distance = np.min(distances_to_private)
            candidates = values[np.isclose(distances_to_private, min_distance, atol=1e-12, rtol=0.0)]
            S[i] = float(np.mean(candidates))
            popular_used[i] = True
            continue

        k_eff = min(kappa, n_values - 1)
        dispersals = np.empty(n_values, dtype=float)
        for idx in range(n_values):
            other_values = np.delete(values, idx)
            distances = np.abs(values[idx] - other_values)
            nearest = np.partition(distances, k_eff - 1)[:k_eff]
            dispersals[idx] = float(np.sum(nearest))

        min_dispersal = np.min(dispersals)
        candidate_values = values[np.isclose(dispersals, min_dispersal, atol=1e-12, rtol=0.0)]
        if len(candidate_values) > 1:
            distances_to_private = np.abs(candidate_values - x[i])
            min_distance = np.min(distances_to_private)
            candidate_values = candidate_values[np.isclose(distances_to_private, min_distance, atol=1e-12, rtol=0.0)]

        S[i] = float(np.mean(candidate_values))
        popular_used[i] = True

    return S, popular_used


def compute_expression_target(x, y_prev, M_prev, params):
    # 统一表达目标入口，确保主模型与附录替代目标仅在 S_i(t) 的定义上不同。
    target = str(params.target).strip().lower()
    if target == "mean":
        return compute_S_mean(y_prev, M_prev), np.zeros(len(y_prev), dtype=bool)
    if target in {"hard_dominant", "hard_dominant_side"}:
        return compute_S_hard_dominant(
            y_prev,
            M_prev,
            tau_S=params.tau_S,
            delta_S=params.delta_S,
            n_dom_min=params.n_dom_min,
        )
    if target in {"popular_cluster", "popular_not_average"}:
        return compute_S_popular_cluster(
            x,
            y_prev,
            M_prev,
            kappa=params.kappa,
        )
    raise ValueError(f"Unknown expression target: {params.target}")

#选择发言与否
def draw_participation(x, c, H, S_mean, params, rng):
    minority_indicator = compute_perceived_minority_indicator(
        x=x,
        S_mean=S_mean,
        definition=params.minority_definition,
    )
    utility = (
        params.theta_0
        + params.theta_M * np.abs(x - 0.5)
        + params.theta_H * H
        - params.theta_C * c * (1.0 - H) ** params.eta
        - params.theta_minor * minority_indicator
    )
    p = sigmoid(utility)
    return rng.binomial(1, p).astype(int), p, minority_indicator

#更新公开观点
def update_expression(x, y_prev, active, S, alpha, sigma_y, rng):
    y = y_prev.copy()
    noise = rng.normal(0.0, sigma_y, size=len(x))
    y_candidate = clip01((1.0 - alpha) * x + alpha * S + noise)
    y[active == 1] = y_candidate[active == 1]
    return y

#参与者发多少帖
def draw_posts(active, y, q, lambda_0, omega, rng):
    m = np.zeros(len(y), dtype=int)
    active_idx = np.where(active == 1)[0]
    if len(active_idx) == 0:
        return m
    lam = lambda_0 * np.exp(omega * np.abs(y[active_idx] - 0.5) + q[active_idx])
    m[active_idx] = 1 + rng.poisson(lam)
    return m


# 供给匹配等率消融：固定当期总发帖量，并在参与者之间等概率分配额外帖子。
def draw_supply_matched_equal_rate_posts(active, target_total_posts, rng):
    m = np.zeros(len(active), dtype=int)
    active_idx = np.where(active == 1)[0]
    if len(active_idx) == 0:
        if int(target_total_posts) != 0:
            raise ValueError(
                f"Cannot match target_total_posts={int(target_total_posts)} with active_count=0."
            )
        return m

    target_total_posts = int(target_total_posts)
    if target_total_posts < len(active_idx):
        raise ValueError(
            f"Cannot match target_total_posts={target_total_posts} below active_count={len(active_idx)}."
        )

    extra_posts = target_total_posts - len(active_idx)
    allocation = rng.multinomial(
        extra_posts,
        np.full(len(active_idx), 1.0 / len(active_idx)),
    )
    m[active_idx] = 1 + allocation
    return m


def draw_posts_with_distribution(active, y, q, params, rng):
    m = np.zeros(len(y), dtype=int)
    active_idx = np.where(active == 1)[0]
    if len(active_idx) == 0:
        return m

    lam = params.lambda_0 * np.exp(params.omega * np.abs(y[active_idx] - 0.5) + q[active_idx])
    distribution = str(params.posting_distribution).strip().lower()
    if distribution == "poisson":
        m[active_idx] = 1 + rng.poisson(lam)
        return m
    if distribution in {"negative_binomial", "nb"}:
        nb_r = params.nb_r
        if nb_r is None or nb_r <= 0:
            raise ValueError("negative_binomial posting requires a positive nb_r")
        p = nb_r / (nb_r + lam)
        m[active_idx] = 1 + rng.negative_binomial(nb_r, p)
        return m
    raise ValueError(f"Unknown posting_distribution: {params.posting_distribution}")


def compute_producer_popularity_signal(v_prev_algorithm, mode):
    mode = str(mode).strip().lower()
    if mode in {"none", "", "disabled"}:
        return np.zeros(len(v_prev_algorithm), dtype=float)
    if mode in {"log_exposure_prev", "algorithm_exposure_prev"}:
        return np.log1p(v_prev_algorithm.astype(float))
    raise ValueError(f"Unknown popularity_signal: {mode}")

#只对有限数值计算平均值，忽略 nan 或无穷值
def finite_nanmean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite))

#初始化模型 创建一批用户，并给每个用户分配初始属性
def initialize_model(config, rngs):
    N = config.simulation.N
    x = draw_private_opinion(
        N=N,
        mean=config.private_opinion.mean,
        sigma=config.private_opinion.sigma_x,
        lower=config.private_opinion.lower,
        upper=config.private_opinion.upper,
        rng=rngs.private_opinion,
        distribution=config.private_opinion.distribution,
        beta_a=config.private_opinion.beta_a,
        beta_b=config.private_opinion.beta_b,
        mixture_weight_left=config.private_opinion.mixture_weight_left,
        left_mean=config.private_opinion.left_mean,
        left_sigma=config.private_opinion.left_sigma,
        right_mean=config.private_opinion.right_mean,
        right_sigma=config.private_opinion.right_sigma,
        empirical_path=config.private_opinion.empirical_path,
        empirical_value_col=config.private_opinion.empirical_value_col,
        empirical_weight_col=config.private_opinion.empirical_weight_col,
    )
    adjacency = generate_network_adjacency(
        config.network.type,
        N,
        config.network.m_BA,
        rngs.network,
        x=x,
        p_ER=config.network.p_ER,
        k_WS=config.network.k_WS,
        p_WS=config.network.p_WS,
        homophily_chi=config.network.homophily_chi,
        attachment_epsilon=config.network.attachment_epsilon,
    )
    k = degree_array(adjacency)
    z = standardized_log_degree(k)
    c = draw_cost(N=N, rng=rngs.cost, low=config.cost.low, high=config.cost.high)
    q = draw_activity_tendency(z=z, sigma_A=config.activity.sigma_A, rho_kq=config.activity.rho_kq, rng=rngs.activity)
    b0, pi = draw_profile(
        x=x,
        lambda_b=config.profile.lambda_b,
        sigma_b=config.profile.sigma_b,
        lambda_pi=config.profile.lambda_pi,
        sigma_pi=config.profile.sigma_pi,
        rng=rngs.profile,
    )
    return {
        "adjacency": adjacency,
        "k": k,
        "z": z,
        "x": x,
        "c": c,
        "q": q,
        "b0": b0,
        "pi": pi,
        "y": b0.copy(),
        "M_prev": [[] for _ in range(N)],
        "P_X": weighted_pairwise_distance(x, np.ones_like(x, dtype=float)),
    }

#决定指标计算时使用哪个 delta_B 如果 config.metrics.delta_B 没有单独设置，
#就用 config.metrics.delta_m；否则用 config.metrics.delta_B。
def metric_delta_B(config):
    return config.metrics.delta_m if config.metrics.delta_B is None else config.metrics.delta_B

#六层分布
def build_layer_distributions(x, y, a, m, v_R, v_V):
    Fx = (x, np.ones_like(x, dtype=float))
    Fa = (x, a.astype(float))
    Fy = (y, a.astype(float))
    Fp = (y, m.astype(float))
    Fr = (y, v_R.astype(float))
    Fv = (y, v_V.astype(float))
    return Fx, Fa, Fy, Fp, Fr, Fv


def initialize_layer_distribution_accumulator(bin_edges):
    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = len(bin_edges) - 1
    return {
        "bin_edges": bin_edges,
        "n_periods": 0,
        "mass_sum": {
            "X": np.zeros(n_bins, dtype=float),
            "A": np.zeros(n_bins, dtype=float),
            "Y": np.zeros(n_bins, dtype=float),
            "P": np.zeros(n_bins, dtype=float),
            "R": np.zeros(n_bins, dtype=float),
            "V": np.zeros(n_bins, dtype=float),
        },
    }


def _normalized_histogram_mass(values, weights, bin_edges):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.zeros(len(bin_edges) - 1, dtype=float)

    values = values[mask]
    weights = weights[mask]
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    mass, _ = np.histogram(values, bins=bin_edges, weights=weights / total)
    return mass.astype(float)


def update_layer_distribution_accumulator(accumulator, Fx, Fa, Fy, Fp, Fr, Fv):
    accumulator["n_periods"] += 1
    layer_map = {
        "X": Fx,
        "A": Fa,
        "Y": Fy,
        "P": Fp,
        "R": Fr,
        "V": Fv,
    }
    for layer_key, (values, weights) in layer_map.items():
        accumulator["mass_sum"][layer_key] += _normalized_histogram_mass(
            values,
            weights,
            accumulator["bin_edges"],
        )


def finalize_layer_distribution_records(accumulator):
    n_periods = int(accumulator["n_periods"])
    if n_periods == 0:
        return []

    bin_edges = accumulator["bin_edges"]
    widths = np.diff(bin_edges)
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    records = []
    for layer_key, mass_sum in accumulator["mass_sum"].items():
        mean_mass = mass_sum / n_periods
        mean_density = mean_mass / widths
        for idx in range(len(mids)):
            records.append(
                {
                    "layer": layer_key,
                    "bin_id": idx,
                    "bin_left": float(bin_edges[idx]),
                    "bin_right": float(bin_edges[idx + 1]),
                    "bin_mid": float(mids[idx]),
                    "mean_mass": float(mean_mass[idx]),
                    "mean_density": float(mean_density[idx]),
                    "window_periods": n_periods,
                }
            )
    return records


def initialize_node_diagnostic_accumulator(N):
    return {
        "n_periods": 0,
        "active_sum": np.zeros(N, dtype=float),
        "posts_sum": np.zeros(N, dtype=float),
        "y_sum": np.zeros(N, dtype=float),
        "abs_y_center_sum": np.zeros(N, dtype=float),
        "v_R_sum": np.zeros(N, dtype=float),
        "v_V_sum": np.zeros(N, dtype=float),
        "v_actual_sum": np.zeros(N, dtype=float),
    }


def update_node_diagnostic_accumulator(accumulator, a, m, y, v_R, v_V, v_actual):
    accumulator["n_periods"] += 1
    accumulator["active_sum"] += a.astype(float)
    accumulator["posts_sum"] += m.astype(float)
    accumulator["y_sum"] += y.astype(float)
    accumulator["abs_y_center_sum"] += np.abs(y - 0.5)
    accumulator["v_R_sum"] += v_R.astype(float)
    accumulator["v_V_sum"] += v_V.astype(float)
    accumulator["v_actual_sum"] += v_actual.astype(float)


def finalize_node_diagnostic_records(state, accumulator):
    n_periods = int(accumulator["n_periods"])
    if n_periods == 0:
        return []

    x = state["x"]
    c = state["c"]
    q = state["q"]
    pi = state["pi"]
    k = state["k"]
    z = state["z"]

    mean_active = accumulator["active_sum"] / n_periods
    mean_posts = accumulator["posts_sum"] / n_periods
    mean_y = accumulator["y_sum"] / n_periods
    mean_abs_y_center = accumulator["abs_y_center_sum"] / n_periods
    mean_v_R = accumulator["v_R_sum"] / n_periods
    mean_v_V = accumulator["v_V_sum"] / n_periods
    mean_v_actual = accumulator["v_actual_sum"] / n_periods
    mean_algorithm_lift = mean_v_V - mean_v_R

    records = []
    for node_id in range(len(x)):
        records.append(
            {
                "node_id": node_id,
                "window_periods": n_periods,
                "x": float(x[node_id]),
                "x_extremity": float(2.0 * abs(x[node_id] - 0.5)),
                "pi": float(pi[node_id]),
                "cost": float(c[node_id]),
                "activity_tendency_q": float(q[node_id]),
                "degree": float(k[node_id]),
                "standardized_log_degree": float(z[node_id]),
                "mean_active": float(mean_active[node_id]),
                "mean_posts": float(mean_posts[node_id]),
                "mean_y": float(mean_y[node_id]),
                "mean_y_extremity": float(2.0 * mean_abs_y_center[node_id]),
                "mean_v_R": float(mean_v_R[node_id]),
                "mean_v_V": float(mean_v_V[node_id]),
                "mean_v_actual": float(mean_v_actual[node_id]),
                "mean_algorithm_lift": float(mean_algorithm_lift[node_id]),
                "total_posts": float(accumulator["posts_sum"][node_id]),
                "total_v_R": float(accumulator["v_R_sum"][node_id]),
                "total_v_V": float(accumulator["v_V_sum"][node_id]),
                "total_v_actual": float(accumulator["v_actual_sum"][node_id]),
                "total_algorithm_lift": float(accumulator["v_V_sum"][node_id] - accumulator["v_R_sum"][node_id]),
            }
        )
    return records


def run_single_model(
    config,
    rngs,
    variant="full",
    full_participation_counts=None,
    full_post_counts=None,
    collect_transitions=False,
    collect_node_diagnostics=False,
    node_diagnostics_start=None,
    collect_layer_distributions=False,
    layer_distribution_start=None,
    layer_distribution_bin_edges=None,
    visibility_intervention_time=None,
    visibility_intervention="random",
):
    state = initialize_model(config, rngs)
    N = config.simulation.N
    T = config.simulation.T
    x = state["x"]
    c = state["c"]
    q = state["q"]
    pi = state["pi"]
    adjacency = state["adjacency"]
    y = state["y"]
    M_prev = state["M_prev"]
    moderate_mask = np.abs(x - 0.5) <= config.metrics.delta_m
    v_prev_algorithm = np.zeros(N, dtype=float)

    if visibility_intervention_time is not None:
        visibility_intervention_time = int(visibility_intervention_time)
        if not 1 <= visibility_intervention_time <= T:
            raise ValueError("visibility_intervention_time must be between 1 and T.")
        visibility_intervention = str(visibility_intervention).strip().lower()
        if visibility_intervention not in {"random", "algorithm"}:
            raise ValueError("visibility_intervention must be 'random' or 'algorithm'.")

    records = []
    participation_counts = []
    post_counts = []
    transition_records = []
    transition_bin_edges = default_bin_edges(config.metrics.transition_bins_Q) if collect_transitions else None
    if collect_node_diagnostics:
        diagnostic_start = T - config.simulation.W + 1 if node_diagnostics_start is None else int(node_diagnostics_start)
        diagnostic_start = max(1, min(T, diagnostic_start))
        node_diagnostic_accumulator = initialize_node_diagnostic_accumulator(N)
    else:
        diagnostic_start = None
        node_diagnostic_accumulator = None
    if collect_layer_distributions:
        layer_start = T - config.simulation.W + 1 if layer_distribution_start is None else int(layer_distribution_start)
        layer_start = max(1, min(T, layer_start))
        if layer_distribution_bin_edges is None:
            layer_distribution_bin_edges = np.linspace(0.0, 1.0, 51)
        layer_distribution_accumulator = initialize_layer_distribution_accumulator(layer_distribution_bin_edges)
    else:
        layer_start = None
        layer_distribution_accumulator = None

    for t in range(1, T + 1):
        matched_supply_target = np.nan
        matched_supply_difference = np.nan
        matched_supply_feasible = np.nan
        H = compute_perceived_congruence(
            x,
            y,
            M_prev,
            definition=getattr(config.participation, "congruence_type", "item"),
        )
        S, target_override_used = compute_expression_target(x, y, M_prev, config.expression)
        S_mean = compute_S_mean(y, M_prev)
        expression_target = str(config.expression.target).strip().lower()

        if variant == "random_selection":
            if full_participation_counts is None:
                raise ValueError("RandomSelection requires full_participation_counts.")
            n_active = int(full_participation_counts[t - 1])
            active_idx = rngs.participation.choice(N, size=n_active, replace=False)
            a = np.zeros(N, dtype=int)
            a[active_idx] = 1
            p = np.full(N, np.nan)
            minority_indicator = np.zeros(N, dtype=float)
        else:
            a, p, minority_indicator = draw_participation(
                x=x,
                c=c,
                H=H,
                S_mean=S_mean,
                params=config.participation,
                rng=rngs.participation,
            )

        if variant == "honest_expression":
            y_new = y.copy()
            y_new[a == 1] = x[a == 1]
            y = y_new
        else:
            y = update_expression(
                x=x,
                y_prev=y,
                active=a,
                S=S,
                alpha=config.expression.alpha,
                sigma_y=config.expression.sigma_y,
                rng=rngs.expression_noise,
            )

        if variant == "equal_posts":
            m = a.astype(int)
        elif variant == "equal_rate":
            m = np.zeros(N, dtype=int)
            active_idx = np.where(a == 1)[0]
            if len(active_idx) > 0:
                m[active_idx] = 1 + rngs.posts.poisson(config.activity.lambda_0, size=len(active_idx))
        elif variant == "matched_equal_rate":
            if full_post_counts is None:
                raise ValueError("MatchedEqualRate requires full_post_counts.")
            # 精确保留同一 replication 的 Full-M0 当期总供给，只消除用户间发帖率异质性。
            matched_supply_target = int(full_post_counts[t - 1])
            m = draw_supply_matched_equal_rate_posts(
                active=a,
                target_total_posts=matched_supply_target,
                rng=rngs.posts,
            )
            matched_supply_difference = int(m.sum()) - matched_supply_target
            matched_supply_feasible = 1.0
        else:
            m = draw_posts_with_distribution(
                active=a,
                y=y,
                q=q,
                params=config.activity,
                rng=rngs.posts,
            )

        D, _candidate_sizes = build_candidate_items(adjacency, m)
        M_R = sample_random_visibility(D=D, K=config.visibility.K, rng=rngs.random_visibility)
        producer_popularity = compute_producer_popularity_signal(
            v_prev_algorithm,
            config.visibility.popularity_signal,
        )
        M_V = sample_algorithm_visibility(
            D=D,
            K=config.visibility.K,
            y=y,
            pi=pi,
            beta_E=config.visibility.beta_E,
            beta_P=config.visibility.beta_P,
            beta_G=config.visibility.beta_G,
            producer_popularity=producer_popularity,
            rng=rngs.algorithm_visibility,
        )
        v_R = exposure_counts(M_R, N)
        v_V = exposure_counts(M_V, N)
        v_prev_algorithm = v_V.astype(float)

        intervention_active = (
            visibility_intervention_time is not None and t >= visibility_intervention_time
        )
        if variant == "random_visibility":
            actual_visibility = "random"
        elif intervention_active:
            actual_visibility = visibility_intervention
        else:
            actual_visibility = config.visibility.actual_visibility
        M_actual = select_actual_visibility(M_R, M_V, actual_visibility)
        v_actual = v_R if actual_visibility == "random" else v_V

        Fx, Fa, Fy, Fp, Fr, Fv = build_layer_distributions(x, y, a, m, v_R, v_V)
        Factual = (y, v_actual.astype(float))
        metric_record, _layer_record = accounting_decomposition(
            Fx,
            Fa,
            Fy,
            Fp,
            Fr,
            Fv,
            delta_e=config.metrics.delta_e,
            delta_m=config.metrics.delta_m,
            delta_B=metric_delta_B(config),
            bpw_p=config.metrics.bpw_p,
        )
        actual_metric_record = distribution_differences(
            Fx,
            Factual,
            delta_e=config.metrics.delta_e,
            delta_m=config.metrics.delta_m,
            delta_B=metric_delta_B(config),
            bpw_p=config.metrics.bpw_p,
        )
        # 同时记录平均感知偏差及其用户间异质性；actual 遵循当前消融的实际反馈口径。
        PRB_R, PRB_var_R, PRB_sd_R, valid_rate_R = observer_prb_stats(M_R, x, y)
        PRB_V, PRB_var_V, PRB_sd_V, valid_rate_V = observer_prb_stats(M_V, x, y)
        # Full-M0 actual visibility is algorithmic, while RandomVisibility actual visibility is random;
        # reuse the already-computed PRB values instead of traversing the same feeds a third time.
        if actual_visibility == "algorithm":
            PRB_actual, PRB_var_actual, PRB_sd_actual, valid_rate_actual = PRB_V, PRB_var_V, PRB_sd_V, valid_rate_V
        elif actual_visibility == "random":
            PRB_actual, PRB_var_actual, PRB_sd_actual, valid_rate_actual = PRB_R, PRB_var_R, PRB_sd_R, valid_rate_R
        else:
            PRB_actual, PRB_var_actual, PRB_sd_actual, valid_rate_actual = observer_prb_stats(M_actual, x, y)

        diag_record = feed_diagnostics(active=a, m=m, D=D, M_R=M_R, M_V=M_V, K=config.visibility.K, v_V=v_V)
        if collect_node_diagnostics and t >= diagnostic_start:
            update_node_diagnostic_accumulator(
                node_diagnostic_accumulator,
                a=a,
                m=m,
                y=y,
                v_R=v_R,
                v_V=v_V,
                v_actual=v_actual,
            )
        if collect_layer_distributions and t >= layer_start:
            update_layer_distribution_accumulator(
                layer_distribution_accumulator,
                Fx=Fx,
                Fa=Fa,
                Fy=Fy,
                Fp=Fp,
                Fr=Fr,
                Fv=Fv,
            )
        if collect_transitions:
            transition_records.extend(transition_records_for_visibility(M_R, x, transition_bin_edges, t, "random"))
            transition_records.extend(transition_records_for_visibility(M_V, x, transition_bin_edges, t, "algorithm"))
            transition_records.extend(transition_records_for_visibility(M_actual, x, transition_bin_edges, t, "actual"))

        record = {
            "t": t,
            **metric_record,
            **actual_metric_record,
            "MeanDegree": float(np.mean(state["k"])),
            "DegreeStd": float(np.std(state["k"])),
            "PRB_R": PRB_R,
            "PRB_V": PRB_V,
            "EPRB": PRB_V - PRB_R,
            "PRB_actual": PRB_actual,
            "EPRB_actual": PRB_actual - PRB_R,
            "PRB_var_R": PRB_var_R,
            "PRB_var_V": PRB_var_V,
            "PRB_var_actual": PRB_var_actual,
            "PRB_sd_R": PRB_sd_R,
            "PRB_sd_V": PRB_sd_V,
            "PRB_sd_actual": PRB_sd_actual,
            "ValidRate_PRB_R": valid_rate_R,
            "ValidRate_PRB_V": valid_rate_V,
            "ValidRate_PRB_actual": valid_rate_actual,
            "H_mean": float(np.mean(H)),
            "S_mean": float(np.mean(S)),
            "MinorityRate": float(np.mean(minority_indicator)),
            "PopularitySignalMean": float(np.mean(producer_popularity)),
            "ExpressionTargetMean": float(np.mean(S)),
            "DominantTargetRate": float(np.mean(target_override_used))
            if expression_target in {"hard_dominant", "hard_dominant_side"}
            else 0.0,
            "PopularTargetRate": float(np.mean(target_override_used))
            if expression_target in {"popular_cluster", "popular_not_average"}
            else 0.0,
            "ParticipationProbMean": finite_nanmean(p),
            "ActiveCount": int(a.sum()),
            "PostsPerParticipant": float(m.sum() / a.sum()) if a.sum() > 0 else float("nan"),
            "MatchedSupplyTarget": matched_supply_target,
            "MatchedSupplyDifference": matched_supply_difference,
            "MatchedSupplyFeasible": matched_supply_feasible,
            "VisibilityInterventionActive": float(intervention_active),
            "ActualVisibilityIsAlgorithm": float(actual_visibility == "algorithm"),
            "ModerateParticipationRate": float(np.mean(a[moderate_mask])) if np.any(moderate_mask) else float("nan"),
            "ClosureError_P": decomposition_closure_error(metric_record, "P"),
            "ClosureError_BPW": decomposition_closure_error(metric_record, "BPW"),
            **diag_record,
        }
        records.append(record)
        participation_counts.append(int(a.sum()))
        post_counts.append(int(m.sum()))
        M_prev = M_actual

    output = {
        "records": records,
        "participation_counts": participation_counts,
        "post_counts": post_counts,
        "transition_records": transition_records,
        "initial_state": state,
        "P_X": state["P_X"],
    }
    if collect_node_diagnostics:
        output["node_diagnostics"] = finalize_node_diagnostic_records(state, node_diagnostic_accumulator)
        output["node_diagnostics_start"] = diagnostic_start
    if collect_layer_distributions:
        output["layer_distribution_records"] = finalize_layer_distribution_records(layer_distribution_accumulator)
        output["layer_distribution_start"] = layer_start
    return output
