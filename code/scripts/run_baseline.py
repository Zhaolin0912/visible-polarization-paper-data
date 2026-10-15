from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m0_lps_epo.config import load_config
from m0_lps_epo.io import save_outputs
from m0_lps_epo.replications import run_replications


REPORT_COLUMNS = [
    "ParticipationRate",
    "TotalPosts",
    "MeanCandidates",
    "MedianCandidates",
    "FeedSaturation",
    "MeanVisibleItems_R",
    "MeanVisibleItems_V",
    "ValidObserverRate_R",
    "ValidObserverRate_V",
    "VRB_P",
    "VRB_BPW",
    "VRB_U_mu",
    "VRB_CON",
    "VRB_BTM",
    "S_deltaB_V",
    "BAL_deltaB_V",
    "S_deltaB_actual",
    "BAL_deltaB_actual",
    "B_selection_P",
    "B_expression_P",
    "B_activity_P",
    "B_reach_P",
    "B_algorithm_P",
    "B_selection_BPW",
    "B_expression_BPW",
    "B_activity_BPW",
    "B_reach_BPW",
    "B_algorithm_BPW",
    "B_selection_U_mu",
    "B_expression_U_mu",
    "B_activity_U_mu",
    "B_reach_U_mu",
    "B_algorithm_U_mu",
    "PRB_R",
    "PRB_V",
    "PRB_actual",
    "PRB_var_R",
    "PRB_var_V",
    "PRB_var_actual",
    "PRB_sd_R",
    "PRB_sd_V",
    "PRB_sd_actual",
    "EPRB",
    "EPRB_actual",
    "AMB",
    "MDI",
    "VMA_10pct",
    "Gini_V",
    "ClosureError_P",
    "ClosureError_BPW",
]


def main():
    config = load_config(ROOT / "configs" / "baseline.yaml")
    # Four independent CPU workers provide a conservative balance of speed and memory use.
    out = run_replications(config, variant="full", show_progress=True, max_workers=4)
    output_dir = ROOT / config.experiment.output_dir
    save_outputs(out, output_dir)

    summary = out["summary"]
    available = [col for col in REPORT_COLUMNS if col in summary.columns]
    report_mean = summary[available].mean().to_frame().T
    report_mean.insert(0, "positive_VRB_BPW_rate", (summary["VRB_BPW"] > 0).mean())
    report_mean.insert(0, "positive_VRB_P_rate", (summary["VRB_P"] > 0).mean())
    report_mean.insert(0, "t_star_mean", summary["t_star"].mean())
    report_mean.insert(0, "convergence_rate", summary["converged"].mean())
    report_sem = summary[available].sem().to_frame().T
    report_ci95 = pd.DataFrame(
        {
            "mean": summary[available].mean(),
            "sem": summary[available].sem(),
        }
    )
    report_ci95["ci95_low"] = report_ci95["mean"] - 1.96 * report_ci95["sem"]
    report_ci95["ci95_high"] = report_ci95["mean"] + 1.96 * report_ci95["sem"]

    report_mean.to_csv(output_dir / "report_mean.csv", index=False)
    report_sem.to_csv(output_dir / "report_sem.csv", index=False)
    report_ci95.to_csv(output_dir / "report_ci95.csv")

    display_cols = [
        "replication_id",
        "converged",
        "t_star",
        "ParticipationRate",
        "TotalPosts",
        "FeedSaturation",
        "VRB_P",
        "VRB_BPW",
        "VRB_U_mu",
        "B_selection_P",
        "B_expression_P",
        "B_activity_P",
        "B_reach_P",
        "B_algorithm_P",
        "PRB_actual",
        "PRB_sd_actual",
        "EPRB_actual",
        "ClosureError_P",
    ]
    print(summary[display_cols].to_string(index=False))
    print("\nFull-scale polarizing algorithm baseline report mean:")
    print(report_mean.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
