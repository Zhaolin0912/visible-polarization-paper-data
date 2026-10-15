from __future__ import annotations

import numpy as np
import pandas as pd


_EMPIRICAL_DISTRIBUTION_CACHE = {}

#鏂囦欢璐熻矗鐢熸垚妯℃嫙鐢ㄦ埛鐨勫熀纭€灞炴€э紝鍖呮嫭鐪熷疄瑙傜偣銆佽〃杈炬垚鏈€佹椿璺冨€惧悜銆佸垵濮嬪叕寮€琛ㄨ揪鍜岀畻娉曞亸濂姐€?
def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


#鎴柇寮忔鎬佸垎甯冩娊鍙?
def _draw_truncnorm(N, mean, sigma, lower, upper, rng):
    if sigma <= 0:
        return np.full(N, mean, dtype=float)

    out = np.empty(N, dtype=float)
    filled = 0
    while filled < N:
        draw_count = max((N - filled) * 2, 32)  # 鑷冲皯32涓紝鎶藉彇闇€瑕佺殑2鍊?
        samples = rng.normal(mean, sigma, size=draw_count)
        accepted = samples[(samples >= lower) & (samples <= upper)]
        take = min(len(accepted), N - filled)
        if take > 0:
            out[filled : filled + take] = accepted[:take]
            filled += take
    return out


def _load_empirical_distribution(path, value_col, weight_col, lower, upper):
    cache_key = (str(path), str(value_col), str(weight_col), float(lower), float(upper))
    if cache_key in _EMPIRICAL_DISTRIBUTION_CACHE:
        return _EMPIRICAL_DISTRIBUTION_CACHE[cache_key]

    frame = pd.read_csv(path)
    if value_col not in frame.columns:
        raise ValueError(f"Empirical opinion file does not contain value column: {value_col}")
    if weight_col not in frame.columns:
        raise ValueError(f"Empirical opinion file does not contain weight column: {weight_col}")

    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values >= lower) & (values <= upper)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("Empirical opinion distribution has no valid support points.")

    probs = weights / np.sum(weights)
    _EMPIRICAL_DISTRIBUTION_CACHE[cache_key] = (values, probs)
    return values, probs


# 新增多种私人意见分布，支持 F_X 形状鲁棒性分析。
def draw_private_opinion(
    N,
    mean,
    sigma,
    lower,
    upper,
    rng,
    distribution="truncnorm",
    beta_a=None,
    beta_b=None,
    mixture_weight_left=0.5,
    left_mean=None,
    left_sigma=None,
    right_mean=None,
    right_sigma=None,
    empirical_path=None,
    empirical_value_col="x",
    empirical_weight_col="weight",
):
    distribution = str(distribution).strip().lower()
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        raise ValueError("upper must be greater than or equal to lower")

    if distribution in {"truncnorm", "truncated_normal"}:
        return _draw_truncnorm(N, mean, sigma, lower, upper, rng)

    if distribution == "uniform":
        if upper == lower:
            return np.full(N, lower, dtype=float)
        return rng.uniform(lower, upper, size=N)

    if distribution == "beta":
        if beta_a is None or beta_b is None:
            raise ValueError("beta distribution requires beta_a and beta_b")
        if beta_a <= 0 or beta_b <= 0:
            raise ValueError("beta_a and beta_b must be positive")
        samples = rng.beta(beta_a, beta_b, size=N)
        return lower + (upper - lower) * samples

    if distribution in {"bimodal", "bimodal_mixture", "two_component_truncnorm"}:
        weight_left = float(np.clip(mixture_weight_left, 0.0, 1.0))
        left_mean = mean if left_mean is None else float(left_mean)
        right_mean = mean if right_mean is None else float(right_mean)
        left_sigma = sigma if left_sigma is None else float(left_sigma)
        right_sigma = sigma if right_sigma is None else float(right_sigma)

        draw_left = rng.binomial(1, weight_left, size=N).astype(bool)
        out = np.empty(N, dtype=float)
        n_left = int(np.sum(draw_left))
        n_right = N - n_left
        if n_left > 0:
            out[draw_left] = _draw_truncnorm(n_left, left_mean, left_sigma, lower, upper, rng)
        if n_right > 0:
            out[~draw_left] = _draw_truncnorm(n_right, right_mean, right_sigma, lower, upper, rng)
        return out

    if distribution in {"empirical", "empirical_weighted", "anes2020"}:
        if empirical_path is None:
            raise ValueError("empirical private opinion distribution requires empirical_path")
        values, probs = _load_empirical_distribution(
            path=empirical_path,
            value_col=empirical_value_col,
            weight_col=empirical_weight_col,
            lower=lower,
            upper=upper,
        )
        return rng.choice(values, size=N, replace=True, p=probs)

    raise ValueError(f"Unknown private opinion distribution: {distribution}")


def draw_cost(N, rng, low=0.0, high=1.0):
    return rng.uniform(low, high, size=N)


def draw_activity_tendency(z, sigma_A, rho_kq, rng):
    rho = float(np.clip(rho_kq, -1.0, 1.0))
    eps = rng.normal(0.0, 1.0, size=len(z))
    q_raw = sigma_A * (rho * z + np.sqrt(max(0.0, 1.0 - rho**2)) * eps)
    correction = np.log(np.mean(np.exp(q_raw)))
    return q_raw - correction


def draw_profile(x, lambda_b, sigma_b, lambda_pi, sigma_pi, rng):
    eps_b = rng.normal(0.0, sigma_b, size=len(x))
    b0 = clip01(0.5 + lambda_b * (x - 0.5) + eps_b)
    eps_pi = rng.normal(0.0, sigma_pi, size=len(x))
    pi = clip01(0.5 + lambda_pi * (b0 - 0.5) + eps_pi)
    return b0, pi
