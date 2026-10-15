from __future__ import annotations #减少一些兼容性问题

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from .convergence import find_convergence_time, stable_average
from .model import run_single_model
from .rng import make_rng_streams
from .transitions import summarize_transition_records

#确定收敛阈值ϵ，判断系统稳定时允许多大波动
def _epsilon_for_run(config, P_X: float) -> float:
    if config.simulation.epsilon_type == "absolute":
        return float(config.simulation.epsilon)
    if config.simulation.epsilon_type == "relative_to_PX":
        return float(config.simulation.epsilon_scale) * float(P_X)
    raise ValueError(f"Unknown epsilon_type: {config.simulation.epsilon_type}")

# 跑一次完整仿真 + 收敛判断 + 稳态汇总
def run_replication(
    config,
    replication_id: int,
    variant="full",
    full_participation_counts=None,
    full_post_counts=None,
    collect_transitions=False,
    visibility_intervention_time=None,
    visibility_intervention="random",
):
    rngs = make_rng_streams(config.experiment.seed, replication_id)
    result = run_single_model(
        config=config,
        rngs=rngs,
        variant=variant,
        full_participation_counts=full_participation_counts,
        full_post_counts=full_post_counts,
        collect_transitions=collect_transitions,
        visibility_intervention_time=visibility_intervention_time,
        visibility_intervention=visibility_intervention,
    )
    df = pd.DataFrame(result["records"])
    epsilon = _epsilon_for_run(config, result["P_X"])
    t_star = find_convergence_time(df=df, W=config.simulation.W, epsilon=epsilon)
    summary = stable_average(df=df, t_star=t_star, W=config.simulation.W)
    summary["replication_id"] = replication_id
    summary["variant"] = variant
    summary["P_X"] = float(result["P_X"])
    summary["epsilon"] = epsilon
    out = {
        "time_series": df,
        "summary": summary,
        "participation_counts": result["participation_counts"],
        "post_counts": result["post_counts"],
    }
    if collect_transitions:
        transition_df = pd.DataFrame(result["transition_records"])
        if not transition_df.empty:
            transition_df["replication_id"] = replication_id
            transition_df["variant"] = variant
        transition_summary = summarize_transition_records(transition_df, t_star=t_star, W=config.simulation.W)
        if not transition_summary.empty:
            transition_summary["replication_id"] = replication_id
            transition_summary["variant"] = variant
        out["transition_time_series"] = transition_df
        out["transition_summary"] = transition_summary
    return out


def _run_replication_worker(
    config,
    replication_id,
    variant,
    full_counts,
    full_post_counts,
    collect_transitions,
    visibility_intervention_time,
    visibility_intervention,
):
    # 顶层 worker 函数可以被 Windows spawn 模式安全序列化。
    start_time = time.time()
    out = run_replication(
        config=config,
        replication_id=replication_id,
        variant=variant,
        full_participation_counts=full_counts,
        full_post_counts=full_post_counts,
        collect_transitions=collect_transitions,
        visibility_intervention_time=visibility_intervention_time,
        visibility_intervention=visibility_intervention,
    )
    return replication_id, out, time.time() - start_time


def _print_replication_progress(variant, completed, total, replication_id, out, replication_elapsed, start_time):
    total_elapsed = time.time() - start_time
    avg_elapsed = total_elapsed / completed
    remaining = total - completed
    eta_seconds = avg_elapsed * remaining
    converged_flag = out["summary"].get("converged", False)
    t_star = out["summary"].get("t_star", float("nan"))
    print(
        f"[{variant}] replication {replication_id + 1}/{total} finished "
        f"({completed}/{total} complete) in {replication_elapsed:.1f}s | "
        f"total {total_elapsed/60:.1f}m | ETA {eta_seconds/60:.1f}m | "
        f"converged={converged_flag} | t_star={t_star}",
        flush=True,
    )


def run_replications(
    config,
    variant="full",
    full_counts_by_rep=None,
    full_post_counts_by_rep=None,
    collect_transitions=False,
    show_progress=False,
    max_workers=1,
    visibility_intervention_time=None,
    visibility_intervention="random",
):
    summaries = []
    time_series = []
    counts_by_rep = {}
    post_counts_by_rep = {}
    transition_time_series = []
    transition_summaries = []
    # Track total runtime so long baseline jobs can print progress and a simple ETA.
    start_time = time.time()

    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    # 先按 replication_id 收集结果，再排序聚合，确保并行与单进程输出行序一致。
    outputs_by_rep = {}
    if max_workers == 1:
        for completed, r in enumerate(range(config.simulation.R), start=1):
            full_counts = None if full_counts_by_rep is None else full_counts_by_rep[r]
            full_post_counts = None if full_post_counts_by_rep is None else full_post_counts_by_rep[r]
            replication_id, out, replication_elapsed = _run_replication_worker(
                config,
                r,
                variant,
                full_counts,
                full_post_counts,
                collect_transitions,
                visibility_intervention_time,
                visibility_intervention,
            )
            outputs_by_rep[replication_id] = out
            if show_progress:
                _print_replication_progress(
                    variant,
                    completed,
                    config.simulation.R,
                    replication_id,
                    out,
                    replication_elapsed,
                    start_time,
                )
    else:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
            futures = []
            for r in range(config.simulation.R):
                full_counts = None if full_counts_by_rep is None else full_counts_by_rep[r]
                full_post_counts = None if full_post_counts_by_rep is None else full_post_counts_by_rep[r]
                futures.append(
                    executor.submit(
                        _run_replication_worker,
                        config,
                        r,
                        variant,
                        full_counts,
                        full_post_counts,
                        collect_transitions,
                        visibility_intervention_time,
                        visibility_intervention,
                    )
                )
            for completed, future in enumerate(as_completed(futures), start=1):
                replication_id, out, replication_elapsed = future.result()
                outputs_by_rep[replication_id] = out
                if show_progress:
                    _print_replication_progress(
                        variant,
                        completed,
                        config.simulation.R,
                        replication_id,
                        out,
                        replication_elapsed,
                        start_time,
                    )

    for r in range(config.simulation.R):
        out = outputs_by_rep[r]
        summaries.append(out["summary"])
        counts_by_rep[r] = out["participation_counts"]
        post_counts_by_rep[r] = out["post_counts"]
        ts = out["time_series"].copy()
        ts["replication_id"] = r
        ts["variant"] = variant
        time_series.append(ts)
        if collect_transitions:
            if "transition_time_series" in out and not out["transition_time_series"].empty:
                transition_time_series.append(out["transition_time_series"])
            if "transition_summary" in out and not out["transition_summary"].empty:
                transition_summaries.append(out["transition_summary"])
    result = {
        "summary": pd.DataFrame(summaries),
        "time_series": pd.concat(time_series, ignore_index=True),
        "participation_counts_by_rep": counts_by_rep,
        "post_counts_by_rep": post_counts_by_rep,
    }
    if collect_transitions:
        result["transition_time_series"] = (
            pd.concat(transition_time_series, ignore_index=True) if transition_time_series else pd.DataFrame()
        )
        result["transition_summary"] = (
            pd.concat(transition_summaries, ignore_index=True) if transition_summaries else pd.DataFrame()
        )
    return result
