from __future__ import annotations

from pathlib import Path


def save_outputs(out, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out["summary"].to_csv(output_dir / "summary.csv", index=False)
    out["time_series"].to_csv(output_dir / "time_series.csv", index=False)
    if "transition_time_series" in out and not out["transition_time_series"].empty:
        out["transition_time_series"].to_csv(output_dir / "transition_time_series.csv", index=False)
    if "transition_summary" in out and not out["transition_summary"].empty:
        out["transition_summary"].to_csv(output_dir / "transition_summary.csv", index=False)
