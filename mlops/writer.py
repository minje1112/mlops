from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_data_ref(data_ref: str) -> None:
    if not data_ref.strip():
        raise ValueError("dataset reference must not be empty")

    parsed = urlparse(data_ref)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise ValueError("http(s) dataset reference must include a host")
        return
    if parsed.scheme in {"s3", "gs"}:
        if not parsed.netloc or not parsed.path:
            raise ValueError("cloud dataset reference must include bucket and path")
        return
    if parsed.scheme == "file":
        if not parsed.path:
            raise ValueError("file dataset reference must include a path")
        return
    if parsed.scheme:
        if len(parsed.scheme) == 1 and data_ref[1:3] in {":\\", ":/"}:
            return
        raise ValueError("dataset reference must use a supported URL/URI scheme")
    if (
        Path(data_ref).expanduser().is_absolute()
        or data_ref.startswith("./")
        or data_ref.startswith("../")
        or Path(data_ref).name
    ):
        return
    raise ValueError("dataset reference must be a URL, URI, or filesystem path")


@dataclass
class MLOpsWriter:
    project_name: str
    objective: str = ""
    created_at: str = field(default_factory=_timestamp)
    actions: list[dict[str, Any]] = field(default_factory=list)
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    training_runs: list[dict[str, Any]] = field(default_factory=list)
    utilities: dict[str, Any] = field(default_factory=dict)

    def add_action(self, title: str, details: str, category: str = "operation") -> dict[str, Any]:
        entry = {
            "timestamp": _timestamp(),
            "title": title,
            "details": details,
            "category": category,
        }
        self.actions.append(entry)
        return entry

    def set_dataset(self, name: str, data_ref: str, note: str = "") -> dict[str, Any]:
        _validate_data_ref(data_ref)
        current = self.datasets.get(name, {"name": name, "changes": []})
        previous_ref = current.get("data_ref")
        current["data_ref"] = data_ref
        current["note"] = note
        current["updated_at"] = _timestamp()
        if previous_ref and previous_ref != data_ref:
            current["changes"].append(
                {
                    "timestamp": _timestamp(),
                    "change_type": "reference_update",
                    "from": previous_ref,
                    "to": data_ref,
                    "details": "Dataset reference changed",
                }
            )
        self.datasets[name] = current
        return current

    def log_dataset_change(self, name: str, change_type: str, details: str) -> dict[str, Any]:
        if name not in self.datasets:
            raise KeyError(f"dataset '{name}' not found")
        now = _timestamp()
        change = {
            "timestamp": now,
            "change_type": change_type,
            "details": details,
        }
        self.datasets[name]["changes"].append(change)
        self.datasets[name]["updated_at"] = now
        return change

    def add_training_run(
        self,
        model_name: str,
        dataset_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        if dataset_name not in self.datasets:
            raise KeyError(f"dataset '{dataset_name}' not found")
        run = {
            "timestamp": _timestamp(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_ref": self.datasets[dataset_name]["data_ref"],
            "metrics": metrics,
            "hyperparameters": hyperparameters or {},
            "notes": notes,
        }
        self.training_runs.append(run)
        return run

    def add_utility(self, name: str, value: Any) -> None:
        self.utilities[name] = value

    def suggest_utilities(self) -> dict[str, list[str]]:
        # Lightweight utilities inspired by experiment-tracking MLOps tools.
        suggestions = {
            "experiment_hygiene": [
                "Define objective, dataset ref, and baseline metric before each run",
                "Track hyperparameters and random seed for reproducibility",
                "Record failure notes for unsuccessful experiments",
            ],
            "release_readiness": [
                "Document offline metric threshold and pass/fail outcome",
                "Link to model card, data quality report, and validation checklist",
                "Record deployment decision and rollback condition",
            ],
        }
        self.utilities["suggested_utilities"] = suggestions
        return suggestions

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    def to_markdown(self, path: str | Path) -> None:
        data = self.as_dict()
        lines = [
            f"# {data['project_name']}",
            "",
            f"- Created: {data['created_at']}",
            f"- Objective: {data['objective'] or 'N/A'}",
            "",
            "## Actions",
        ]
        for action in data["actions"]:
            lines.append(f"- [{action['timestamp']}] **{action['title']}** ({action['category']}): {action['details']}")
        lines += ["", "## Datasets"]
        for dataset_name, dataset in data["datasets"].items():
            lines.append(f"- **{dataset_name}**: {dataset['data_ref']}")
            if dataset.get("note"):
                lines.append(f"  - note: {dataset['note']}")
            for change in dataset["changes"]:
                lines.append(
                    f"  - [{change['timestamp']}] {change['change_type']}: {change['details']}"
                )
        lines += ["", "## Training Runs"]
        for run in data["training_runs"]:
            lines.append(
                f"- [{run['timestamp']}] model={run['model_name']} dataset={run['dataset_name']} "
                f"metrics={run['metrics']}"
            )
        lines += ["", "## Utilities"]
        for key, value in data["utilities"].items():
            lines.append(f"- **{key}**: {value}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
