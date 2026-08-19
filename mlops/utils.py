from __future__ import annotations

from statistics import mean
from typing import Any


def summarize_metrics(training_runs: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    values = [run["metrics"][metric_name] for run in training_runs if metric_name in run.get("metrics", {})]
    if not values:
        return {"count": 0.0, "best": 0.0, "average": 0.0}
    return {"count": float(len(values)), "best": max(values), "average": mean(values)}


def next_experiment_id(existing_experiments: list[dict[str, Any]]) -> str:
    return f"exp-{len(existing_experiments) + 1:03d}"
