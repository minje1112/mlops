from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields as dataclasses_fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

from . import nodes as nodes_mod
from .metrics import evaluate
from .nodes import ARTIFACT_KINDS, DATASET_SPLITS, EXPERIMENT_MODES, EXPERIMENT_STATUSES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _environment() -> dict[str, str]:
    """Metadata captured automatically with every run, for reproducibility."""
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    commit = _git_commit()
    if commit:
        environment["git_commit"] = commit
    return environment


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


def attachment_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    return "file"


# --------------------------------------------------------------------- legacy

def legacy_to_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a pre-node document into the node graph, losslessly.

    The earliest action becomes the root; datasets, runs and artifacts keep the
    lineage that was implied by ``derived_from`` and ``run_id`` and fall back to
    the root when nothing else is known.
    """
    actions = sorted(data.get("actions", []), key=lambda item: item.get("timestamp", ""))
    datasets = data.get("datasets", {})
    runs = data.get("training_runs", [])
    artifacts = data.get("artifacts", [])
    if not (actions or datasets or runs or artifacts):
        return []

    nodes: list[dict[str, Any]] = []
    created = data.get("created_at", _timestamp())

    def add(node_id: str, node_type: str, title: str, summary: str, parents: list[str],
            payload: dict[str, Any], created_at: str, updated_at: str) -> dict[str, Any]:
        node = {
            "id": node_id,
            "type": node_type,
            "title": title or node_id,
            "summary": summary,
            "parents": parents,
            "created_at": created_at or created,
            "updated_at": updated_at or created_at or created,
            "attachments": [],
            "data": payload,
        }
        nodes.append(node)
        return node

    if actions:
        first = actions[0]
        root = add(
            first.get("id") or "act-001", "action", first.get("title", "Project start"),
            first.get("details", ""), [], {"category": first.get("category", "operation")},
            first.get("timestamp", created), first.get("updated_at", ""),
        )
        rest = actions[1:]
    else:
        root = add("act-001", "action", "Project start", "Imported from an earlier document.",
                   [], {"category": "operation"}, created, created)
        rest = []
    root_id = root["id"]

    for index, action in enumerate(rest, start=2):
        add(action.get("id") or f"act-{index:03d}", "action", action.get("title", "action"),
            action.get("details", ""), [root_id], {"category": action.get("category", "operation")},
            action.get("timestamp", created), action.get("updated_at", ""))

    dataset_ids: dict[str, str] = {}
    for index, (name, dataset) in enumerate(datasets.items(), start=1):
        dataset_ids[name] = f"ds-{index:03d}"
    for name, dataset in datasets.items():
        parents = [dataset_ids[parent] for parent in dataset.get("derived_from", []) if parent in dataset_ids]
        add(dataset_ids[name], "dataset", name, dataset.get("note", ""), parents or [root_id],
            {
                "name": name,
                "version": 1,
                "split": "train",
                "data_ref": dataset.get("data_ref", ""),
                "changes": list(dataset.get("changes", [])),
            },
            dataset.get("updated_at", created), dataset.get("updated_at", ""))

    run_ids: dict[str, str] = {}
    for index, run in enumerate(runs, start=1):
        run_id = run.get("run_id") or f"exp-{index:03d}"
        run_ids[run_id] = run_id
        dataset_id = dataset_ids.get(run.get("dataset_name", ""))
        add(run_id, "experiment", run.get("model_name", "run"), run.get("notes", ""),
            [dataset_id] if dataset_id else [root_id],
            {
                "mode": "training",
                "model_name": run.get("model_name", ""),
                "architecture": run.get("architecture", ""),
                "parameters": dict(run.get("parameters") or run.get("hyperparameters") or {}),
                "metrics": dict(run.get("metrics", {})),
                "status": run.get("status", "completed"),
                "duration_seconds": run.get("duration_seconds", 0.0),
                "environment": dict(run.get("environment", {})),
                "error": run.get("error", ""),
                "dataset_roles": {dataset_id: "train"} if dataset_id else {},
            },
            run.get("timestamp", created), run.get("completed_at", ""))

    artifact_ids = {artifact["name"]: f"art-{index:03d}" for index, artifact in enumerate(artifacts, start=1)}
    for artifact in artifacts:
        parents = []
        if artifact.get("run_id") in run_ids:
            parents.append(artifact["run_id"])
        parents += [artifact_ids[parent] for parent in artifact.get("derived_from", []) if parent in artifact_ids]
        add(artifact_ids[artifact["name"]], "artifact", artifact["name"], artifact.get("note", ""),
            parents or [root_id],
            {"artifact_ref": artifact.get("artifact_ref", ""), "kind": artifact.get("kind", "model")},
            artifact.get("timestamp", created), artifact.get("updated_at", ""))

    return nodes


def _repair(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise a loaded graph: known parents only, one root, no orphans."""
    known = {node["id"] for node in nodes}
    for node in nodes:
        node.setdefault("summary", "")
        node.setdefault("attachments", [])
        node.setdefault("data", {})
        node.setdefault("updated_at", node.get("created_at", ""))
        node["parents"] = [parent for parent in node.get("parents", []) if parent in known and parent != node["id"]]
    root_id = next((node["id"] for node in nodes if not node["parents"]), None)
    for node in nodes:
        if not node["parents"] and node["id"] != root_id:
            node["parents"] = [root_id]
    return nodes


class RunLogger:
    """Handle yielded by :meth:`MLOpsWriter.run` for logging a training run."""

    def __init__(self, writer: "MLOpsWriter", node: dict[str, Any]) -> None:
        self._writer = writer
        self.node = node

    @property
    def run_id(self) -> str:
        return self.node["id"]

    @property
    def record(self) -> dict[str, Any]:
        """The experiment payload (parameters, metrics, status, ...)."""
        return self.node["data"]

    def _touch(self) -> None:
        self.node["updated_at"] = _timestamp()

    def log_params(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Record hyperparameters, learning rate, or any other run setting."""
        self.node["data"]["parameters"].update(params or {})
        self.node["data"]["parameters"].update(kwargs)
        self._touch()
        return self.node["data"]["parameters"]

    def set_architecture(self, architecture: str) -> None:
        """Record the architecture choice behind this run."""
        self.node["data"]["architecture"] = architecture
        self._touch()

    def log_metrics(self, metrics: dict[str, float] | None = None, **kwargs: Any) -> dict[str, float]:
        """Record evaluation metrics as they become available.

        Metrics are merged, so results measured later add to what is there
        rather than replacing it.
        """
        self.node["data"]["metrics"].update(metrics or {})
        self.node["data"]["metrics"].update(kwargs)
        self._touch()
        return self.node["data"]["metrics"]

    def log_evaluation(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        *,
        prefix: str = "",
        positive_label: Any | None = None,
        average: str | None = None,
    ) -> dict[str, float]:
        """Compute accuracy/precision/recall/f1 from predictions and log them."""
        scores = evaluate(y_true, y_pred, positive_label=positive_label, average=average)
        self.log_metrics({f"{prefix}{name}": value for name, value in scores.items()})
        return scores

    def log_artifact(
        self,
        name: str,
        artifact_ref: str,
        kind: str = "model",
        note: str = "",
        derived_from: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Register an artifact produced by this run (model file, export, report)."""
        return self._writer.add_artifact(
            name,
            artifact_ref,
            kind=kind,
            run_id=self.run_id,
            note=note,
            derived_from=derived_from,
        )

    def attach_link(self, url: str, label: str = "") -> dict[str, Any]:
        return self._writer.attach_link(self.node, url, label)

    def attach_path(self, path: str, label: str = "") -> dict[str, Any]:
        return self._writer.attach_path(self.node, path, label)

    def attach_file(self, source_path: str | Path, label: str = "") -> dict[str, Any]:
        return self._writer.attach_file(self.node, source_path, label)


@dataclass
class MLOpsWriter:
    project_name: str
    objective: str = ""
    created_at: str = field(default_factory=_timestamp)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    utilities: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Folder this project was last saved to/loaded from; enables run autosave.
        self._root: str | Path | None = None
        _repair(self.nodes)

    # -------------------------------------------------------------- selectors

    @property
    def root(self) -> dict[str, Any] | None:
        """The origin node every other node descends from."""
        return next((node for node in self.nodes if not node.get("parents")), None)

    def nodes_of_type(self, node_type: str) -> list[dict[str, Any]]:
        return [node for node in self.nodes if node.get("type") == node_type]

    @property
    def action_nodes(self) -> list[dict[str, Any]]:
        return self.nodes_of_type("action")

    @property
    def note_nodes(self) -> list[dict[str, Any]]:
        return self.nodes_of_type("note")

    @property
    def dataset_versions(self) -> list[dict[str, Any]]:
        return self.nodes_of_type("dataset")

    @property
    def experiments(self) -> list[dict[str, Any]]:
        return self.nodes_of_type("experiment")

    @property
    def artifact_nodes(self) -> list[dict[str, Any]]:
        return self.nodes_of_type("artifact")

    def get_node(self, node: str | dict[str, Any]) -> dict[str, Any]:
        """Look a node up by id, or pass one straight through."""
        if isinstance(node, dict):
            return node
        return nodes_mod.require(self.nodes, node)

    def ancestors(self, node: str | dict[str, Any]) -> list[dict[str, Any]]:
        """Where this node came from, nearest ancestors last."""
        found = nodes_mod.ancestors(self.nodes, self.get_node(node)["id"])
        return [item for item in self.nodes if item["id"] in found]

    def children(self, node: str | dict[str, Any]) -> list[dict[str, Any]]:
        node_id = self.get_node(node)["id"]
        return [item for item in self.nodes if node_id in item.get("parents", [])]

    def graph(self) -> dict[str, Any]:
        return nodes_mod.graph(self.nodes)

    # Kept so existing callers and exports keep working.
    lineage = graph

    # ------------------------------------------------------------------ nodes

    def _resolve_parents(self, parents: str | Sequence[str] | None) -> list[str]:
        """``None`` attaches to the project root; anything else is taken as given."""
        if parents is not None:
            return [self.get_node(parent)["id"] if isinstance(parent, dict) else parent
                    for parent in nodes_mod.as_list(parents)]
        root = self.root
        return [root["id"]] if root else []

    def add_node(
        self,
        node_type: str,
        title: str,
        parents: str | Sequence[str] = (),
        summary: str = "",
        **data: Any,
    ) -> dict[str, Any]:
        """Add a node. Only the first node in a project may omit parents."""
        node = nodes_mod.new_node(
            node_type, title, self.nodes, nodes_mod.as_list(parents), summary, data
        )
        self.nodes.append(node)
        return node

    def edit_node(
        self,
        node: str | dict[str, Any],
        title: str | None = None,
        summary: str | None = None,
        parents: Sequence[str] | None = None,
        **data: Any,
    ) -> dict[str, Any]:
        """Edit a node in place and stamp its last-edited time."""
        entry = self.get_node(node)
        if title is None and summary is None and parents is None and not data:
            raise ValueError("edit_node needs at least one field to change")
        if title is not None:
            if not title.strip():
                raise ValueError("node title must not be empty")
            entry["title"] = title.strip()
        if summary is not None:
            entry["summary"] = summary
        if parents is not None:
            entry["parents"] = nodes_mod.validate_parents(
                self.nodes, nodes_mod.as_list(parents), node_id=entry["id"]
            )
        if data:
            self._validate_data(entry["type"], data)
            entry["data"].update(data)
        entry["updated_at"] = _timestamp()
        return entry

    def _validate_data(self, node_type: str, data: dict[str, Any]) -> None:
        if node_type == "dataset":
            if "split" in data and data["split"] not in DATASET_SPLITS:
                raise ValueError(f"dataset split must be one of {', '.join(DATASET_SPLITS)}")
            if "data_ref" in data:
                _validate_data_ref(data["data_ref"])
        elif node_type == "experiment":
            if "mode" in data and data["mode"] not in EXPERIMENT_MODES:
                raise ValueError(f"experiment mode must be one of {', '.join(EXPERIMENT_MODES)}")
            if "status" in data and data["status"] not in EXPERIMENT_STATUSES:
                raise ValueError(f"experiment status must be one of {', '.join(EXPERIMENT_STATUSES)}")
        elif node_type == "artifact":
            if "kind" in data and data["kind"] not in ARTIFACT_KINDS:
                raise ValueError(f"artifact kind must be one of {', '.join(ARTIFACT_KINDS)}")
            if "artifact_ref" in data:
                _validate_data_ref(data["artifact_ref"])

    # ---------------------------------------------------------------- actions

    def add_action(
        self,
        title: str,
        details: str = "",
        category: str = "operation",
        parents: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Record something you did. The first one becomes the project root."""
        return self.add_node(
            "action", title, self._resolve_parents(parents), summary=details, category=category
        )

    def add_note(
        self,
        title: str,
        details: str = "",
        parents: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Record an observation hanging off the work that prompted it."""
        return self.add_node("note", title, self._resolve_parents(parents), summary=details)

    def get_action(self, action: str | int) -> dict[str, Any]:
        """Look an action up by its id (``act-001``) or list position."""
        if isinstance(action, int):
            try:
                return self.action_nodes[action]
            except IndexError:
                raise KeyError(f"action index {action} is out of range") from None
        return nodes_mod.require(self.nodes, action)

    def edit_action(
        self,
        action: str | int,
        title: str | None = None,
        details: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Edit an action in place and stamp its last-edited time."""
        entry = self.get_action(action)
        if title is None and details is None and category is None:
            raise ValueError("edit_action needs at least one of title, details, category")
        data = {"category": category} if category is not None else {}
        return self.edit_node(entry, title=title, summary=details, **data)

    @staticmethod
    def action_was_edited(action: dict[str, Any]) -> bool:
        return nodes_mod.was_edited(action)

    # --------------------------------------------------------------- datasets

    def latest_dataset(self, name: str) -> dict[str, Any] | None:
        """The newest version node for a dataset name."""
        versions = [node for node in self.dataset_versions if node["data"].get("name") == name]
        return max(versions, key=lambda node: node["data"].get("version", 0), default=None)

    def dataset_history(self, name: str) -> list[dict[str, Any]]:
        versions = [node for node in self.dataset_versions if node["data"].get("name") == name]
        return sorted(versions, key=lambda node: node["data"].get("version", 0))

    def add_dataset(
        self,
        name: str,
        data_ref: str,
        split: str = "train",
        note: str = "",
        parents: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Register the next version of a dataset.

        Versions chain: the previous version is always a parent of the new one,
        so the history of a dataset is visible in the graph.
        """
        _validate_data_ref(data_ref)
        if split not in DATASET_SPLITS:
            raise ValueError(f"dataset split must be one of {', '.join(DATASET_SPLITS)}")
        previous = self.latest_dataset(name)
        version = previous["data"]["version"] + 1 if previous else 1

        if parents is None and previous is not None:
            resolved = [previous["id"]]
        else:
            resolved = self._resolve_parents(parents)
            if previous is not None and previous["id"] not in resolved:
                resolved = [previous["id"], *resolved]

        return self.add_node(
            "dataset", name, resolved, summary=note,
            name=name, version=version, split=split, data_ref=data_ref, changes=[],
        )

    def set_dataset(
        self,
        name: str,
        data_ref: str,
        note: str = "",
        derived_from: str | Sequence[str] | None = None,
        split: str = "train",
    ) -> dict[str, Any]:
        """Point a dataset at a reference, versioning it when the reference changes."""
        previous = self.latest_dataset(name)
        parents = None
        if derived_from is not None:
            parents = [
                (self.latest_dataset(parent) or {}).get("id", parent)
                for parent in nodes_mod.as_list(derived_from)
            ]
            if name in nodes_mod.as_list(derived_from):
                raise ValueError("a dataset cannot be derived from itself")

        if previous is not None and previous["data"]["data_ref"] == data_ref:
            updates: dict[str, Any] = {}
            if parents:
                merged = previous["parents"] + [p for p in parents if p not in previous["parents"]]
                updates["parents"] = merged
            return self.edit_node(previous, summary=note, **updates)

        return self.add_dataset(
            name, data_ref, split=previous["data"]["split"] if previous else split,
            note=note, parents=parents,
        )

    def log_dataset_change(self, name: str, change_type: str, details: str) -> dict[str, Any]:
        """Append to the change history of a dataset's newest version."""
        node = self.latest_dataset(name)
        if node is None:
            raise KeyError(f"dataset '{name}' not found")
        change = {"timestamp": _timestamp(), "change_type": change_type, "details": details}
        node["data"].setdefault("changes", []).append(change)
        node["updated_at"] = change["timestamp"]
        return change

    # ------------------------------------------------------------ experiments

    def _dataset_parents(
        self,
        dataset: str | None,
        datasets: dict[str, str] | None,
    ) -> tuple[list[str], dict[str, str]]:
        """Resolve dataset names or node ids to parents plus their roles."""
        wanted: dict[str, str] = {}
        if dataset is not None:
            wanted[dataset] = "train"
        for key, role in (datasets or {}).items():
            wanted[key] = role

        parents: list[str] = []
        roles: dict[str, str] = {}
        for key, role in wanted.items():
            node = self.latest_dataset(key) or nodes_mod.find(self.nodes, key)
            if node is None or node["type"] != "dataset":
                raise KeyError(f"dataset '{key}' not found")
            if node["id"] not in parents:
                parents.append(node["id"])
            roles[node["id"]] = role
        return parents, roles

    def _new_experiment(
        self,
        model_name: str,
        dataset: str | None,
        datasets: dict[str, str] | None,
        parents: str | Sequence[str] | None,
        mode: str,
        architecture: str,
        parameters: dict[str, Any] | None,
        notes: str,
        status: str,
        title: str | None,
    ) -> dict[str, Any]:
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f"experiment mode must be one of {', '.join(EXPERIMENT_MODES)}")
        dataset_parents, roles = self._dataset_parents(dataset, datasets)
        # Omitting parents falls back to the root; passing an empty list is an
        # explicit "no origin", which the graph rules reject.
        if parents is None:
            extra = [] if dataset_parents else self._resolve_parents(None)
        else:
            extra = self._resolve_parents(parents)
        merged = dataset_parents + [p for p in extra if p not in dataset_parents]

        return self.add_node(
            "experiment", title or model_name, merged, summary=notes,
            mode=mode,
            model_name=model_name,
            architecture=architecture,
            parameters=dict(parameters or {}),
            metrics={},
            status=status,
            duration_seconds=0.0,
            environment=_environment(),
            error="",
            dataset_roles=roles,
        )

    def log_into(self, node: str | dict[str, Any]) -> RunLogger:
        """Re-open a finished experiment to add parameters, metrics or artifacts.

        Returns the same handle :meth:`run` yields, so results measured after
        training — a later evaluation, a checkpoint you scored the next day —
        merge into the run instead of replacing what is already recorded.

        >>> run = writer.log_into("exp-001")
        >>> run.log_metrics(test_tag_acc=0.91)
        >>> writer.save()
        """
        entry = self.get_node(node)
        if entry["type"] != "experiment":
            raise ValueError(f"{entry['id']} is a {entry['type']} node, not an experiment")
        entry["data"].setdefault("parameters", {})
        entry["data"].setdefault("metrics", {})
        return RunLogger(self, entry)

    def add_experiment(
        self,
        model_name: str,
        parents: str | Sequence[str] | None = None,
        mode: str = "evaluation",
        dataset: str | None = None,
        datasets: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        architecture: str = "",
        notes: str = "",
        title: str | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        """Record an experiment you already have results for."""
        if status not in EXPERIMENT_STATUSES:
            raise ValueError(f"experiment status must be one of {', '.join(EXPERIMENT_STATUSES)}")
        node = self._new_experiment(
            model_name, dataset, datasets, parents, mode, architecture,
            parameters, notes, status, title,
        )
        node["data"]["metrics"] = dict(metrics or {})
        return node

    def add_training_run(
        self,
        model_name: str,
        dataset_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any] | None = None,
        notes: str = "",
        parameters: dict[str, Any] | None = None,
        architecture: str = "",
        parents: str | Sequence[str] | None = None,
        mode: str = "training",
    ) -> dict[str, Any]:
        """Record a finished run. ``hyperparameters`` and ``parameters`` are merged."""
        merged = {**(hyperparameters or {}), **(parameters or {})}
        node = self._new_experiment(
            model_name, dataset_name, None, parents, mode, architecture,
            merged, notes, "completed", None,
        )
        node["data"]["metrics"] = dict(metrics)
        return node

    @contextmanager
    def run(
        self,
        model_name: str,
        dataset: str | None = None,
        parameters: dict[str, Any] | None = None,
        architecture: str = "",
        notes: str = "",
        autosave: bool | None = None,
        parents: str | Sequence[str] | None = None,
        datasets: dict[str, str] | None = None,
        mode: str = "training",
        title: str | None = None,
    ) -> Iterator[RunLogger]:
        """Track a training run, capturing timing, status and environment automatically.

        >>> with writer.run("lightgbm", "train", parameters={"learning_rate": 0.05}) as run:
        ...     run.log_metrics(pr_auc=0.71)

        The dataset given by name becomes a parent of the run, so every
        experiment records the data it came from. Pass ``parents`` to add the
        earlier experiments or actions this one builds on.
        """
        node = self._new_experiment(
            model_name, dataset, datasets, parents, mode, architecture,
            parameters, notes, "running", title,
        )
        started = time.perf_counter()
        try:
            yield RunLogger(self, node)
        except BaseException as error:
            node["data"]["status"] = "failed"
            node["data"]["error"] = f"{type(error).__name__}: {error}"
            raise
        else:
            node["data"]["status"] = "completed"
        finally:
            node["updated_at"] = _timestamp()
            node["data"]["duration_seconds"] = round(time.perf_counter() - started, 3)
            if autosave or (autosave is None and self._root is not None):
                self.save(self._root)

    # -------------------------------------------------------------- artifacts

    def add_artifact(
        self,
        name: str,
        artifact_ref: str,
        kind: str = "model",
        run_id: str | None = None,
        note: str = "",
        derived_from: str | Sequence[str] | None = None,
        parents: str | Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Register a deployable output and where it came from."""
        _validate_data_ref(artifact_ref)
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"artifact kind must be one of {', '.join(ARTIFACT_KINDS)}")

        resolved: list[str] = []
        if run_id:
            run = nodes_mod.find(self.nodes, run_id)
            if run is None or run["type"] != "experiment":
                raise KeyError(f"training run '{run_id}' not found")
            resolved.append(run_id)
        for parent in nodes_mod.as_list(derived_from):
            if parent == name:
                raise ValueError("an artifact cannot be derived from itself")
            match = next((node for node in self.artifact_nodes if node["title"] == parent), None)
            found = match["id"] if match else parent
            if nodes_mod.find(self.nodes, found) is None:
                raise KeyError(f"artifact '{parent}' not found")
            if found not in resolved:
                resolved.append(found)
        for parent in self._resolve_parents(parents) if parents is not None else []:
            if parent not in resolved:
                resolved.append(parent)
        if not resolved and parents is None:
            resolved = self._resolve_parents(None)

        existing = next((node for node in self.artifact_nodes if node["title"] == name), None)
        if existing is not None:
            merged = existing["parents"] + [p for p in resolved if p not in existing["parents"] and p != existing["id"]]
            updates: dict[str, Any] = {"artifact_ref": artifact_ref, "kind": kind}
            return self.edit_node(
                existing, summary=note or existing["summary"], parents=merged or None, **updates
            )

        return self.add_node(
            "artifact", name, resolved, summary=note, artifact_ref=artifact_ref, kind=kind,
        )

    # ------------------------------------------------------------ attachments

    def _attach(self, node: str | dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        entry = self.get_node(node)
        record["id"] = nodes_mod.next_attachment_id(entry)
        record["added_at"] = _timestamp()
        entry.setdefault("attachments", []).append(record)
        entry["updated_at"] = record["added_at"]
        return record

    def attach_link(self, node: str | dict[str, Any], url: str, label: str = "") -> dict[str, Any]:
        """Attach a hyperlink (paper, dashboard, model card) to a node."""
        if not urlparse(url).scheme:
            raise ValueError("link attachment must be a URL with a scheme")
        return self._attach(node, {"type": "link", "label": label or url, "href": url})

    def attach_path(self, node: str | dict[str, Any], path: str | Path, label: str = "") -> dict[str, Any]:
        """Attach a filesystem path, left where it is and openable from the dashboard."""
        text = str(path).strip()
        if not text:
            raise ValueError("path attachment must not be empty")
        return self._attach(node, {"type": "path", "label": label or text, "href": text})

    def attach_file(
        self,
        node: str | dict[str, Any],
        source_path: str | Path,
        label: str = "",
        root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Copy a picture, PDF or other file into the docs folder and attach it."""
        from .store import project_slug, store_attachment

        entry = self.get_node(node)
        source = Path(source_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"no file at {source}")
        data = source.read_bytes()
        stored = store_attachment(
            project_slug(self.project_name), entry["id"], source.name, data,
            root if root is not None else self._root,
        )
        return self.attach_stored(entry, stored.name, len(data), label or source.name)

    def attach_stored(
        self,
        node: str | dict[str, Any],
        filename: str,
        size: int,
        label: str = "",
    ) -> dict[str, Any]:
        """Attach a file already written into the docs folder (used by uploads)."""
        return self._attach(node, {
            "type": attachment_type_for(filename),
            "label": label or filename,
            "filename": filename,
            "size": size,
            "href": "",
        })

    def remove_attachment(self, node: str | dict[str, Any], attachment_id: str) -> dict[str, Any]:
        """Detach a file or link; stored copies are deleted from the docs folder."""
        from .store import delete_attachment, project_slug

        entry = self.get_node(node)
        attachments = entry.get("attachments", [])
        match = next((item for item in attachments if item.get("id") == attachment_id), None)
        if match is None:
            raise KeyError(f"attachment '{attachment_id}' not found on {entry['id']}")
        if match.get("filename"):
            delete_attachment(project_slug(self.project_name), entry["id"], match["filename"], self._root)
        attachments.remove(match)
        entry["updated_at"] = _timestamp()
        return match

    # ---------------------------------------------------------------- utility

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

    # ------------------------------------------------------------------- i/o

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLOpsWriter":
        payload = dict(data)
        if "nodes" not in payload:
            payload["nodes"] = legacy_to_nodes(payload)
        fields = {f.name for f in dataclasses_fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in fields})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    def _tree_lines(self) -> list[str]:
        """The graph as an indented outline, deepest branch first."""
        depths = nodes_mod.depths(self.nodes)
        by_id = {node["id"]: node for node in self.nodes}
        children: dict[str, list[str]] = {node["id"]: [] for node in self.nodes}
        for node in self.nodes:
            for parent in node.get("parents", []):
                if parent in children:
                    children[parent].append(node["id"])

        lines: list[str] = []
        seen: set[str] = set()

        def walk(node_id: str, indent: int) -> None:
            node = by_id[node_id]
            label = f"[{node['id']}] {node['type']}: {node['title']}"
            if node_id in seen:
                lines.append(f"{'  ' * indent}- {label} (also above)")
                return
            seen.add(node_id)
            extra = ""
            if node["type"] == "dataset":
                extra = f" — v{node['data'].get('version', 1)} {node['data'].get('split', '')}"
            elif node["type"] == "experiment":
                extra = f" — {node['data'].get('mode', '')} / {node['data'].get('status', '')}"
            lines.append(f"{'  ' * indent}- {label}{extra}")
            for child in sorted(children[node_id], key=lambda item: depths.get(item, 0)):
                walk(child, indent + 1)

        for node in self.nodes:
            if not node.get("parents"):
                walk(node["id"], 0)
        for node in self.nodes:  # anything unreachable still gets listed
            if node["id"] not in seen:
                walk(node["id"], 0)
        return lines

    def to_markdown(self, path: str | Path) -> None:
        lines = [
            f"# {self.project_name}",
            "",
            f"- Created: {self.created_at}",
            f"- Objective: {self.objective or 'N/A'}",
            "",
            "## Tree",
            *self._tree_lines(),
            "",
            "## Actions",
        ]
        for action in self.action_nodes + self.note_nodes:
            edited = f" (edited {action['updated_at']})" if nodes_mod.was_edited(action) else ""
            category = action["data"].get("category", action["type"])
            lines.append(
                f"- [{action['created_at']}] **{action['title']}** ({category}): "
                f"{action['summary']}{edited}"
            )
        lines += ["", "## Datasets"]
        for dataset in self.dataset_versions:
            payload = dataset["data"]
            lines.append(
                f"- **{payload['name']}** v{payload['version']} ({payload['split']}): {payload['data_ref']}"
            )
            if dataset["summary"]:
                lines.append(f"  - note: {dataset['summary']}")
            for change in payload.get("changes", []):
                lines.append(f"  - [{change['timestamp']}] {change['change_type']}: {change['details']}")
        lines += ["", "## Training Runs"]
        for run in self.experiments:
            payload = run["data"]
            lines.append(
                f"- [{run['id']}] {payload['status']} {payload['mode']} model={payload['model_name']} "
                f"metrics={payload['metrics']}"
            )
            if payload.get("parameters"):
                lines.append(f"  - parameters: {payload['parameters']}")
            if payload.get("architecture"):
                lines.append(f"  - architecture: {payload['architecture']}")
            if payload.get("environment"):
                lines.append(f"  - environment: {payload['environment']}")
        lines += ["", "## Artifacts"]
        for artifact in self.artifact_nodes:
            lines.append(
                f"- **{artifact['title']}** ({artifact['data']['kind']}): {artifact['data']['artifact_ref']}"
            )
        lines += ["", "## Attachments"]
        for node in self.nodes:
            for attachment in node.get("attachments", []):
                target = attachment.get("href") or attachment.get("filename", "")
                lines.append(f"- [{node['id']}] {attachment['type']}: {attachment['label']} — {target}")
        lines += ["", "## Lineage"]
        for edge in self.graph()["edges"]:
            lines.append(f"- {edge['from']} --> {edge['to']}")
        lines += ["", "## Utilities"]
        for key, value in self.utilities.items():
            lines.append(f"- **{key}**: {value}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save(self, root: str | Path | None = None) -> Path:
        """Save this project into the folder the web dashboard reads.

        Defaults to ``$MLOPS_DOCS`` when set, otherwise ``./mlops_docs``.
        """
        from .store import resolve_root, save_project

        path = save_project(self, root)
        self._root = resolve_root(root)
        return path
