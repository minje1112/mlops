from __future__ import annotations

import re
from statistics import mean
from typing import Any

# Error rates and losses are better when they are smaller.
LOWER_IS_BETTER = {
    "loss", "error", "err", "cer", "wer", "mae", "mse", "rmse", "mape",
    "perplexity", "ppl", "regret", "latency", "drift", "dist", "distance",
}


def metric_is_lower_better(metric_name: str) -> bool:
    """Whether a smaller value of this metric is the better result."""
    return any(token in LOWER_IS_BETTER for token in re.split(r"[^a-z0-9]+", metric_name.lower()))


def summarize_metrics(training_runs: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    values = [run["metrics"][metric_name] for run in training_runs if metric_name in run.get("metrics", {})]
    if not values:
        return {"count": 0.0, "best": 0.0, "average": 0.0}
    best = min(values) if metric_is_lower_better(metric_name) else max(values)
    return {"count": float(len(values)), "best": best, "average": mean(values)}


def next_experiment_id(existing_experiments: list[dict[str, Any]]) -> str:
    return f"exp-{len(existing_experiments) + 1:03d}"
