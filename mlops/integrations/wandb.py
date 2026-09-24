"""Import a Weights & Biases run into an experiment node.

Two sources, same result:

* a **local run directory** (``wandb/run-20260902_111641-niqajvlo``) — reads the
  JSON files W&B already wrote, so it needs no network and no ``wandb`` package;
* a **W&B run path** (``entity/project/run_id``, or the run's URL) — uses the
  ``wandb`` package, which is only imported when that path is taken.

A crashed run is the interesting case: training records only the exception, while
W&B still holds every metric logged before the crash. Importing recovers them.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

# Throughput bookkeeping the Trainer logs alongside real results. Averaging
# "best total_flos" across runs is meaningless, so these are dropped by default.
THROUGHPUT = re.compile(r"(runtime|samples_per_second|steps_per_second|total_flos)$")

# W&B run states -> the tracker's experiment statuses
STATE_TO_STATUS = {
    "finished": "completed",
    "failed": "failed",
    "crashed": "failed",
    "killed": "failed",
    "running": "running",
    "preempted": "failed",
}


class RunData:
    """Everything worth keeping from a W&B run, source-independent."""

    def __init__(self, **fields: Any) -> None:
        self.id: str = fields.get("id", "")
        self.name: str = fields.get("name", "")
        self.entity: str = fields.get("entity", "")
        self.project: str = fields.get("project", "")
        self.url: str = fields.get("url", "")
        self.state: str = fields.get("state", "")
        self.summary: dict[str, Any] = fields.get("summary", {})
        self.config: dict[str, Any] = fields.get("config", {})
        self.metadata: dict[str, Any] = fields.get("metadata", {})
        self.notes: str = fields.get("notes", "")
        self.error: str = fields.get("error", "")

    # ------------------------------------------------------------------ mapping

    @property
    def status(self) -> str | None:
        """The tracker status, or None when the source does not know the state.

        A local run directory has no authoritative final state, so it is only
        reported as failed when the console log ends in a traceback.
        """
        return STATE_TO_STATUS.get(self.state)

    def metrics(self, keep_throughput: bool = False) -> dict[str, float]:
        """Summary values, minus W&B's own bookkeeping and throughput counters."""
        return {
            name: value
            for name, value in self.summary.items()
            if not name.startswith("_") and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (keep_throughput or not THROUGHPUT.search(name))
        }

    def duration(self) -> float:
        runtime = self.summary.get("_runtime")
        return round(float(runtime), 3) if isinstance(runtime, (int, float)) else 0.0

    def parameters(self, include_config: bool = False) -> dict[str, Any]:
        """The command line that started the run, optionally plus its full config.

        The CLI arguments are what a person actually chose; a Hugging Face
        ``TrainingArguments`` dump adds ~200 derived keys on top, so it is opt-in.
        """
        params = argv_to_params(self.metadata.get("args", []))
        if include_config:
            for key, value in self.config.items():
                if not key.startswith("_"):
                    params.setdefault(key, value)
        return params

    def environment(self) -> dict[str, str]:
        meta, env = self.metadata, {}
        if meta.get("python"):
            env["python"] = str(meta["python"])
        if meta.get("os"):
            env["platform"] = str(meta["os"])
        commit = (meta.get("git") or {}).get("commit", "")
        if commit:
            env["git_commit"] = commit[:7]
        if meta.get("gpu"):
            count = meta.get("gpu_count") or 1
            env["gpu"] = f"{count}x {meta['gpu']}" if count and count > 1 else str(meta["gpu"])
        if meta.get("cudaVersion"):
            env["cuda"] = str(meta["cudaVersion"])
        if meta.get("host"):
            env["host"] = str(meta["host"])
        return env


def argv_to_params(args: list[str]) -> dict[str, Any]:
    """``['--lr', '2e-5', '--lora']`` -> ``{'lr': 2e-05, 'lora': True}``."""
    params: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = str(args[index])
        if not token.startswith("--"):
            index += 1
            continue
        key = token.lstrip("-").replace("-", "_")
        if index + 1 < len(args) and not str(args[index + 1]).startswith("--"):
            params[key] = _typed(str(args[index + 1]))
            index += 2
        else:                      # a bare flag
            params[key] = True
            index += 1
    return params


def _typed(raw: str) -> Any:
    try:
        return json.loads(raw)     # numbers, true/false/null
    except (json.JSONDecodeError, ValueError):
        return raw


# ------------------------------------------------------------------- sources

def from_local(run_dir: str | Path) -> RunData:
    """Read a run straight out of its ``wandb/run-*`` directory."""
    files = Path(run_dir).expanduser()
    if (files / "files").is_dir():
        files = files / "files"
    metadata_path = files / "wandb-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"no wandb-metadata.json under {run_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary_path = files / "wandb-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    # run-20260902_111641-niqajvlo -> niqajvlo
    run_id = files.parent.name.rsplit("-", 1)[-1]
    error = _tail_error(files / "output.log")
    args = argv_to_params(metadata.get("args", []))
    return RunData(
        id=run_id,
        name=str(args.get("wandb_run_name") or Path(metadata.get("program", "")).stem or run_id),
        project=str(args.get("wandb_project", "")),
        summary=summary,
        metadata=metadata,
        error=error,
        # A local directory holds no authoritative final state; a traceback at
        # the end of the console log is the one thing it does prove.
        state="crashed" if error else "",
    )


def _tail_error(log_path: Path, window: int = 8000) -> str:
    """The exception a run died on, read from the tail of its console log."""
    try:
        tail = log_path.read_bytes()[-window:].decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [line.rstrip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return ""
    # W&B captures stdout, so the "Traceback (most recent call last):" header
    # (stderr) is usually absent — but the exception line itself is the last
    # thing the process wrote, which is the signal that matters.
    last = lines[-1]
    if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b\s*(:|$)", last):
        return last[:500]
    return ""


def from_api(run_path: str) -> RunData:
    """Read a run through the W&B API. ``entity/project/run_id`` or a run URL."""
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "importing from the W&B API needs the 'wandb' package; "
            "install it, or point at a local wandb/run-* directory instead"
        ) from error

    path = _normalise_path(run_path)
    try:
        run = wandb.Api().run(path)
    except Exception as error:                 # wandb raises its own CommError
        raise LookupError(
            f"W&B has no run at '{path}' (read from {run_path!r}) — check the "
            "entity, which for a team run is the team and not your username"
        ) from error
    return _from_api_run(run)


def _from_api_run(run: Any) -> RunData:
    """Map a ``wandb.apis.public.Run`` onto RunData (kept pure for testing)."""
    summary = dict(getattr(run, "summary", {}) or {})
    try:
        metadata = dict(run.metadata or {})
    except Exception:                                  # metadata is optional
        metadata = {}
    return RunData(
        id=getattr(run, "id", ""),
        name=getattr(run, "name", "") or getattr(run, "id", ""),
        entity=getattr(run, "entity", ""),
        project=getattr(run, "project", ""),
        url=getattr(run, "url", ""),
        state=getattr(run, "state", ""),
        summary=summary,
        config=dict(getattr(run, "config", {}) or {}),
        metadata=metadata,
        notes=getattr(run, "notes", "") or "",
    )


def _normalise_path(run_path: str) -> str:
    """Reduce however a run was referenced down to ``entity/project/run_id``.

    Accepts the API form, the full run URL, and the URL path people copy out of
    the address bar — with or without a scheme, host or leading slash:

    >>> _normalise_path("/acme-team/Docling-Restart/runs/atnafe6d")
    'acme-team/Docling-Restart/atnafe6d'
    """
    text = str(run_path).strip()
    if not text:
        raise ValueError("no run path given")
    if "://" in text:
        text = urlparse(text).path
    else:
        text = text.split("?", 1)[0].split("#", 1)[0]

    segments = [segment for segment in text.split("/") if segment]
    if segments and "." in segments[0]:          # a bare host, e.g. wandb.ai/...
        segments = segments[1:]
    # entity/project/runs/<id>; positional so a project *named* "runs" survives
    if len(segments) > 3 and segments[2] == "runs":
        del segments[2]
    if len(segments) < 3:
        raise ValueError(
            f"cannot read 'entity/project/run_id' out of {run_path!r} — pass that "
            "form, or the run's URL"
        )
    return "/".join(segments[:3])


# wandb names its directories run-<YYYYMMDD>_<HHMMSS>-<id>, which nothing in a
# W&B run path ever looks like.
LOCAL_RUN_DIR = re.compile(r"(offline-)?run-\d{8}_\d{6}-\w+")


def looks_local(source: str) -> bool:
    """Whether ``source`` is meant as a directory on this machine."""
    text = str(source)
    return bool(
        LOCAL_RUN_DIR.search(text)
        or text.startswith(("~", "./", "../"))
        or "/wandb/" in text
    )


def load(source: str | Path) -> RunData:
    """Pick the local or API source based on what ``source`` looks like."""
    path = Path(str(source)).expanduser()
    if path.exists():
        return from_local(path)
    if looks_local(str(source)):
        # Falling through to the API here would parse a filesystem path as
        # entity/project/run_id and fail somewhere confusing.
        raise FileNotFoundError(
            f"no wandb run directory at {path} — if that run lives on W&B rather "
            "than this machine, pass 'entity/project/run_id' or the run's URL"
        )
    return from_api(str(source))


# --------------------------------------------------------------------- import

def import_run(
    writer: Any,
    source: str | Path,
    parents: str | list[str],
    *,
    mode: str | None = None,
    title: str | None = None,
    model_name: str | None = None,
    include_config: bool = False,
    datasets: dict[str, str] | None = None,
    status: str | None = None,
    keep_throughput: bool = False,
) -> dict[str, Any]:
    """Create (or refresh) the experiment node for a W&B run.

    Re-importing the same run updates its node instead of adding a second one,
    so a run can be pulled again once it finishes.
    """
    return import_run_data(
        writer, load(source), parents, mode=mode, title=title, model_name=model_name,
        include_config=include_config, datasets=datasets, status=status,
        keep_throughput=keep_throughput,
    )


def import_run_data(
    writer: Any,
    run: RunData,
    parents: str | list[str],
    *,
    mode: str | None = None,
    title: str | None = None,
    model_name: str | None = None,
    include_config: bool = False,
    datasets: dict[str, str] | None = None,
    status: str | None = None,
    keep_throughput: bool = False,
) -> dict[str, Any]:
    """Write an already-loaded :class:`RunData` into the project as a node."""
    status = status or run.status
    params = run.parameters(include_config=include_config)
    resolved_mode = mode or ("finetune" if params.get("lora") else "training")
    model = model_name or params.get("model_id") or run.name
    label = title or params.get("output_dir") or run.name or run.id

    existing = next(
        (node for node in writer.experiments if node["data"].get("wandb", {}).get("id") == run.id),
        None,
    )
    payload = {
        "model_name": model,
        "mode": resolved_mode,
        "metrics": run.metrics(keep_throughput),
        "parameters": params,
        "duration_seconds": run.duration(),
        "wandb": {
            "id": run.id, "name": run.name, "entity": run.entity,
            "project": run.project, "url": run.url, "state": run.state,
        },
    }
    environment = run.environment()
    if environment:
        payload["environment"] = environment
    # Only assert a status the source actually knows, so re-importing never
    # overwrites a status the training script recorded first-hand.
    if status is not None:
        payload["status"] = status
    if run.error:
        payload["error"] = run.error

    if existing is not None:
        node = writer.edit_node(existing, title=label, **payload)
    else:
        node = writer.add_experiment(
            model, parents=parents, mode=resolved_mode, datasets=datasets,
            metrics=payload["metrics"], parameters=params,
            notes=run.notes or f"imported from W&B run {run.name or run.id}",
            title=label, status=status or "completed",
        )
        # add_experiment covers model/mode/status/metrics/parameters; the rest
        # of the payload (duration, environment, error, wandb ids) lands here.
        already_set = {"model_name", "mode", "status", "metrics", "parameters"}
        extra = {key: value for key, value in payload.items() if key not in already_set}
        if extra:
            writer.edit_node(node, **extra)

    if run.url and not any(a.get("href") == run.url for a in node.get("attachments", [])):
        writer.attach_link(node, run.url, f"W&B: {run.name or run.id}")
    return node


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..store import load_project

    parser = argparse.ArgumentParser(description="Import a W&B run into an MLOps project node.")
    parser.add_argument("source", help="local wandb/run-* directory, entity/project/run_id, or a run URL")
    parser.add_argument("--project", required=True, help="MLOps project slug")
    parser.add_argument("--parents", required=True, help="comma-separated parent node ids")
    parser.add_argument("--mode", default=None, help="evaluation | training | finetune")
    parser.add_argument("--title", default=None)
    parser.add_argument("--root", default=None, help="docs folder (default: $MLOPS_DOCS)")
    parser.add_argument("--include-config", action="store_true",
                        help="also record the run's full config, not just its command line")
    parser.add_argument("--keep-throughput", action="store_true",
                        help="also record runtime / samples_per_second / total_flos counters")
    args = parser.parse_args(argv)

    writer = load_project(args.project, args.root)
    node = import_run(
        writer, args.source, [p.strip() for p in args.parents.split(",") if p.strip()],
        mode=args.mode, title=args.title, include_config=args.include_config,
        keep_throughput=args.keep_throughput,
    )
    writer.save(args.root)
    print(f"{node['id']}: {node['title']} ({node['data']['status']}, "
          f"{len(node['data']['metrics'])} metrics, {len(node['data']['parameters'])} parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
