#把模型中的私人观点、参与者、公开表达、帖子、随机曝光和算法曝光这些分布，
#转换成一整套可分析的偏差、极化、平衡和感知指标
from __future__ import annotations

from typing import Dict

import numpy as np

#清洗 values 和 weights
def valid_values_weights(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[mask], weights[mask]

#加权平均观点
def weighted_mean(values, weights):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(values * weights) / total)

#加权两两距离，也就是一种成对距离
def weighted_pairwise_distance(values, weights):
    values, weights = valid_values_weights(values, weights)
    total_weight = weights.sum()
    if len(values) == 0 or total_weight <= 0:
        return np.nan

    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cum_w = np.cumsum(w) #到当前位置为止，累计权重是多少。从左至右累加
    cum_wx = np.cumsum(w * x)

    total = 0.0
    for i in range(1, len(x)):
        total += w[i] * (x[i] * cum_w[i - 1] - cum_wx[i - 1])

    return float(2.0 * total / (total_weight**2))

#离中心点 0.5 的平均距离
def weighted_center_distance(values, weights):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(weights * np.abs(values - 0.5)) / total)

#极端观点比例
def weighted_extreme_share(values, weights, delta_e):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(weights * (np.abs(values - 0.5) > delta_e)) / total)

#温和观点比例
def weighted_moderate_share(values, weights, delta_m):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.sum(weights * (np.abs(values - 0.5) < delta_m)) / total)

#把权重标准化成总和为 1
def normalize_weights(values, weights):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return values, weights
    return values, weights / total

#一维 Wasserstein 距离
def weighted_wasserstein_1d(values_a, weights_a, values_b, weights_b):
    xa, wa = normalize_weights(values_a, weights_a)
    xb, wb = normalize_weights(values_b, weights_b)
    if len(xa) == 0 or len(xb) == 0:
        return np.nan

    order_a = np.argsort(xa)#按照观点值从小到大排序
    order_b = np.argsort(xb)
    xa, wa = xa[order_a], wa[order_a]
    xb, wb = xb[order_b], wb[order_b]
    
    #找出两个分布里所有出现过的观点值，然后排序、去重。
    grid = np.sort(np.unique(np.concatenate([xa, xb])))
    if len(grid) <= 1:
        return 0.0
    
    #CDF 的离散版本
    cum_a = np.cumsum(wa)
    cum_b = np.cumsum(wb)
    #对每个 grid 点，找出在 xa 中有多少个值小于等于这个点。
    idx_a = np.searchsorted(xa, grid, side="right")
    idx_b = np.searchsorted(xb, grid, side="right")
    #得到每个 grid 点上的 CDF
    Fa = np.where(idx_a > 0, cum_a[idx_a - 1], 0.0)
    Fb = np.where(idx_b > 0, cum_b[idx_b - 1], 0.0)
    return float(np.sum(np.abs(Fa[:-1] - Fb[:-1]) * np.diff(grid)))


def _equal_reference_prefix_cache(reference_values):
    # PRB_i repeatedly compares a short feed distribution against the same full private-opinion distribution.
    # Precomputing the reference CDF antiderivative avoids allocating a 5000-point grid for every observer.
    ref = np.asarray(reference_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return None

    ref = np.sort(ref)
    n = len(ref)
    boundaries = np.concatenate(([0.0], ref, [1.0]))
    levels = np.arange(n + 1, dtype=float) / n
    lengths = np.diff(boundaries)
    prefix_cache = {}

    def prefix_for(level_numerator, level_denominator):
        key = (int(level_numerator), int(level_denominator))
        if key not in prefix_cache:
            cdf_level = key[0] / key[1]
            increments = lengths * np.abs(cdf_level - levels)
            prefix_cache[key] = np.concatenate(([0.0], np.cumsum(increments)))
        return prefix_cache[key]

    def antiderivative(z, level_numerator, level_denominator):
        z = float(np.clip(z, 0.0, 1.0))
        idx = int(np.searchsorted(boundaries, z, side="right") - 1)
        idx = min(max(idx, 0), len(levels) - 1)
        prefix = prefix_for(level_numerator, level_denominator)
        cdf_level = level_numerator / level_denominator
        return float(prefix[idx] + (z - boundaries[idx]) * abs(cdf_level - levels[idx]))

    return antiderivative


def _wasserstein_to_equal_reference(values, reference_antiderivative):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or reference_antiderivative is None:
        return np.nan

    values = np.sort(np.clip(values, 0.0, 1.0))
    m = len(values)
    bounds = np.concatenate(([0.0], values, [1.0]))
    total = 0.0
    for i in range(m + 1):
        left = bounds[i]
        right = bounds[i + 1]
        if right <= left:
            continue
        total += reference_antiderivative(right, i, m) - reference_antiderivative(left, i, m)
    return float(total)

#p 阶 Wasserstein 距离
def weighted_wasserstein_1d_p(values_a, weights_a, values_b, weights_b, p=1.0):
    if p <= 0:
        raise ValueError("p must be positive.")

    xa, wa = normalize_weights(values_a, weights_a)
    xb, wb = normalize_weights(values_b, weights_b)
    if len(xa) == 0 or len(xb) == 0:
        return np.nan

    order_a = np.argsort(xa)
    order_b = np.argsort(xb)
    xa, wa = xa[order_a], wa[order_a].copy()
    xb, wb = xb[order_b], wb[order_b].copy()

    i = 0
    j = 0
    acc = 0.0
    rem_a = wa[0]
    rem_b = wb[0]

    while i < len(xa) and j < len(xb):
        mass = min(rem_a, rem_b)
        acc += mass * abs(xa[i] - xb[j]) ** p
        rem_a -= mass
        rem_b -= mass

        if rem_a <= 1e-15:
            i += 1
            if i < len(xa):
                rem_a = wa[i]
        if rem_b <= 1e-15:
            j += 1
            if j < len(xb):
                rem_b = wb[j]

    return float(acc ** (1.0 / p))

#左右两侧质量和平衡度
def side_balance_metrics(values, weights, delta_B):
    values, weights = valid_values_weights(values, weights)
    total = weights.sum()
    if total <= 0:
        return {
            "L_deltaB": np.nan,
            "R_deltaB": np.nan,
            "S_deltaB": np.nan,
            "BAL_deltaB": np.nan,
            "BTM": np.nan,
        }

    left = float(np.sum(weights * (values < 0.5 - delta_B)) / total)
    right = float(np.sum(weights * (values > 0.5 + delta_B)) / total)
    side_mass = left + right
    balance = 0.0 if side_mass == 0.0 else 2.0 * min(left, right) / side_mass
    two_sided_mass = 2.0 * min(left, right)
    return {
        "L_deltaB": left,
        "R_deltaB": right,
        "S_deltaB": side_mass,
        "BAL_deltaB": balance,
        "BTM": two_sided_mass,
    }

#单边平均偏向
def weighted_unipolar_mean_bias(values, weights):
    mu = weighted_mean(values, weights)
    if not np.isfinite(mu):
        return np.nan
    return float(2.0 * abs(mu - 0.5))

#单极化指标-共识程度
def weighted_consensus(values, weights):
    pairwise = weighted_pairwise_distance(values, weights)
    if not np.isfinite(pairwise):
        return np.nan
    return float(1.0 - 2.0 * pairwise)

#双极化指标--计算与“两端均衡分布”的距离，并转换成分数
def weighted_bpw(values, weights, p=1.0):
    wdist = weighted_wasserstein_1d_p(values, weights, [0.0, 1.0], [0.5, 0.5], p=p)
    if not np.isfinite(wdist):
        return np.nan
    score = 1.0 - (2.0 ** (1.0 / p)) * wdist
    return float(np.clip(score, 0.0, 1.0))

#汇总一个分布的所有指标
def distribution_metrics(values, weights, delta_e, delta_m, delta_B=None, bpw_p=1.0):
    if delta_B is None:
        delta_B = delta_m
    side_metrics = side_balance_metrics(values, weights, delta_B)
    return {
        "mu": weighted_mean(values, weights),
        "P": weighted_pairwise_distance(values, weights),
        "U_mu": weighted_unipolar_mean_bias(values, weights),
        "CON": weighted_consensus(values, weights),
        "BPW": weighted_bpw(values, weights, p=bpw_p),
        "C": weighted_center_distance(values, weights),
        "E": weighted_extreme_share(values, weights, delta_e),
        "Mod": weighted_moderate_share(values, weights, delta_m),
        **side_metrics,
    }

#把可见偏差分解到不同机制
def accounting_decomposition(Fx, Fa, Fy, Fp, Fr, Fv, delta_e, delta_m, delta_B=None, bpw_p=1.0):
    layers = {
        "X": distribution_metrics(*Fx, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
        "A": distribution_metrics(*Fa, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
        "Y": distribution_metrics(*Fy, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
        "P_layer": distribution_metrics(*Fp, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
        "R": distribution_metrics(*Fr, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
        "V": distribution_metrics(*Fv, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p),
    }

    out = {}
    for g in ["mu", "P", "U_mu", "CON", "BPW", "BTM", "C", "E", "Mod"]:
        out[f"VRB_{g}"] = layers["V"][g] - layers["X"][g]
        out[f"B_selection_{g}"] = layers["A"][g] - layers["X"][g]
        out[f"B_expression_{g}"] = layers["Y"][g] - layers["A"][g]
        out[f"B_activity_{g}"] = layers["P_layer"][g] - layers["Y"][g]
        out[f"B_reach_{g}"] = layers["R"][g] - layers["P_layer"][g]
        out[f"B_algorithm_{g}"] = layers["V"][g] - layers["R"][g]

    for layer_name, layer_metrics in layers.items():
        for g in ["L_deltaB", "R_deltaB", "S_deltaB", "BAL_deltaB", "BTM"]:
            out[f"{g}_{layer_name}"] = layer_metrics[g]

    out["AMB"] = abs(layers["V"]["mu"] - layers["X"]["mu"])
    out["EAMB"] = abs(layers["V"]["mu"] - layers["X"]["mu"]) - abs(layers["R"]["mu"] - layers["X"]["mu"])
    out["MDI"] = layers["X"]["Mod"] - layers["V"]["Mod"]
    return out, layers

#真实分布 vs 实际可见分布
def distribution_differences(Fx, Factual, delta_e, delta_m, suffix="_actual", delta_B=None, bpw_p=1.0):
    x_metrics = distribution_metrics(*Fx, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p)
    actual_metrics = distribution_metrics(*Factual, delta_e, delta_m, delta_B=delta_B, bpw_p=bpw_p)
    out = {}
    for g in ["mu", "P", "U_mu", "CON", "BPW", "BTM", "C", "E", "Mod"]:
        out[f"VRB_{g}{suffix}"] = actual_metrics[g] - x_metrics[g]
    for g in ["L_deltaB", "R_deltaB", "S_deltaB", "BAL_deltaB", "BTM"]:
        out[f"{g}{suffix}"] = actual_metrics[g]
    out[f"AMB{suffix}"] = abs(actual_metrics["mu"] - x_metrics["mu"])
    out[f"MDI{suffix}"] = x_metrics["Mod"] - actual_metrics["Mod"]
    return out

#观察者感知偏差 当前观察者看到的信息流观点分布vs全体用户真实私人观点分布
def observer_prb_stats(M, x, y):
    # 同时汇总个体 PRB_i，避免为均值、方差和标准差重复遍历观察者 feed。
    distances = []
    reference_antiderivative = _equal_reference_prefix_cache(x)
    for items in M:
        if len(items) == 0:
            continue
        #取出观察者看到的公开观点；把当前观察者看到的帖子列表items转换成他看到的公开观点分布 values
        values = np.array([y[producer] for producer, _item_id in items], dtype=float)
        distances.append(_wasserstein_to_equal_reference(values, reference_antiderivative))
    if not distances:
        return np.nan, np.nan, np.nan, 0.0
    distances = np.asarray(distances, dtype=float)
    return (
        float(np.nanmean(distances)),
        float(np.nanvar(distances)),
        float(np.nanstd(distances)),
        len(distances) / len(M),
    )


def observer_prb(M, x, y):
    # 保留原有二元返回接口，供已有脚本或外部调用继续使用。
    mean_prb, _var_prb, _sd_prb, valid_rate = observer_prb_stats(M, x, y)
    return mean_prb, valid_rate


def decomposition_closure_error(metrics: Dict[str, float], g: str = "P") -> float:
    parts = (
        metrics[f"B_selection_{g}"]
        + metrics[f"B_expression_{g}"]
        + metrics[f"B_activity_{g}"]
        + metrics[f"B_reach_{g}"]
        + metrics[f"B_algorithm_{g}"]
    )
    return float(abs(metrics[f"VRB_{g}"] - parts))
