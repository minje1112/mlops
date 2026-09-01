from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .writer import MLOpsWriter

ROOT_ENV_VAR = "MLOPS_DOCS"
DEFAULT_ROOT_NAME = "mlops_docs"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def default_root() -> Path:
    """Where projects live: ``$MLOPS_DOCS`` if set, else ``./mlops_docs``.

    Point ``MLOPS_DOCS`` at one shared folder (e.g. ``~/mlops_docs``) to track
    projects from several repositories in a single dashboard.
    """
    configured = os.environ.get(ROOT_ENV_VAR, "").strip()
    return Path(configured).expanduser() if configured else Path(DEFAULT_ROOT_NAME)


def resolve_root(root: str | Path | None) -> Path:
    return default_root() if root is None else Path(root).expanduser()


def project_slug(project_name: str) -> str:
    """Turn a project name into a filesystem-safe, URL-safe identifier."""
    slug = _SLUG_STRIP.sub("-", project_name.strip().lower()).strip("-")
    return slug or "project"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) and "project_name" in data else None


def _documents(root: str | Path | None) -> list[tuple[Path, dict[str, Any]]]:
    root_path = resolve_root(root)
    if not root_path.is_dir():
        return []
    documents = []
    for path in sorted(root_path.glob("*.json")):
        data = _read(path)
        if data is not None:
            documents.append((path, data))
    return documents


def _target_path(root: Path, project_name: str) -> Path:
    """Reuse the file already holding this project, else claim a free slug."""
    for path, data in _documents(root):
        if data.get("project_name") == project_name:
            return path
    slug = project_slug(project_name)
    candidate = root / f"{slug}.json"
    counter = 2
    while candidate.exists():
        candidate = root / f"{slug}-{counter}.json"
        counter += 1
    return candidate


def save_project(writer: "MLOpsWriter", root: str | Path | None = None) -> Path:
    """Persist a writer as JSON under ``root`` so the web dashboard can read it."""
    root_path = resolve_root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    path = _target_path(root_path, writer.project_name)
    writer.to_json(path)
    return path


def last_activity(document: dict[str, Any]) -> str:
    """Most recent timestamp anywhere in the document (ISO-8601 sorts lexically)."""
    stamps = [document.get("created_at", "")]
    for node in document.get("nodes", []):
        stamps += [node.get("created_at", ""), node.get("updated_at", "")]
        stamps += [item.get("added_at", "") for item in node.get("attachments", [])]
        stamps += [change.get("timestamp", "") for change in node.get("data", {}).get("changes", [])]
    # Documents written before the node graph still list activity correctly.
    for action in document.get("actions", []):
        stamps += [action.get("timestamp", ""), action.get("updated_at", "")]
    for run in document.get("training_runs", []):
        stamps += [run.get("timestamp", ""), run.get("completed_at", "")]
    for artifact in document.get("artifacts", []):
        stamps.append(artifact.get("updated_at", ""))
    for dataset in document.get("datasets", {}).values():
        stamps.append(dataset.get("updated_at", ""))
        stamps += [change.get("timestamp", "") for change in dataset.get("changes", [])]
    return max((stamp for stamp in stamps if stamp), default="")


def summarize_project(slug: str, document: dict[str, Any]) -> dict[str, Any]:
    """Compact record used by the dashboard's project list."""
    from .nodes import counts_by_type
    from .writer import legacy_to_nodes

    nodes = document.get("nodes")
    if nodes is None:  # legacy document, not yet re-saved in node form
        nodes = legacy_to_nodes(document)
    counts = counts_by_type(nodes)
    return {
        "slug": slug,
        "project_name": document.get("project_name", slug),
        "objective": document.get("objective", ""),
        "created_at": document.get("created_at", ""),
        "last_activity": last_activity(document),
        "counts": {**counts, "nodes": len(nodes)},
    }


def list_projects(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Summaries of every saved project, most recently active first."""
    summaries = [summarize_project(path.stem, data) for path, data in _documents(root)]
    summaries.sort(key=lambda item: (item["last_activity"], item["project_name"]), reverse=True)
    return summaries


def get_project(slug: str, root: str | Path | None = None) -> dict[str, Any] | None:
    """Full document for one slug, or ``None`` when it is not stored under ``root``."""
    for path, data in _documents(root):
        if path.stem == slug:
            return data
    return None


def load_project(slug: str, root: str | Path | None = None) -> "MLOpsWriter":
    """Rehydrate a saved project into a writer so tracking can continue."""
    from .writer import MLOpsWriter

    document = get_project(slug, root)
    if document is None:
        raise KeyError(f"project '{slug}' not found in {resolve_root(root)}")
    writer = MLOpsWriter.from_dict(document)
    writer._root = resolve_root(root)
    return writer


def create_project(
    project_name: str,
    objective: str = "",
    root: str | Path | None = None,
) -> "MLOpsWriter":
    """Start a new project and save it straight away.

    Raises ``FileExistsError`` when a project of that name is already tracked.
    """
    from .writer import MLOpsWriter

    if not project_name.strip():
        raise ValueError("project name must not be empty")
    for _, data in _documents(root):
        if data.get("project_name") == project_name:
            raise FileExistsError(f"project '{project_name}' already exists")
    writer = MLOpsWriter(project_name=project_name.strip(), objective=objective.strip())
    writer.save(root)
    return writer


def project_path(slug: str, root: str | Path | None = None) -> Path | None:
    """Path of the file backing ``slug``, or ``None`` when it is not stored."""
    for path, _ in _documents(root):
        if path.stem == slug:
            return path
    return None


# ------------------------------------------------------------------ attachments

ATTACHMENTS_DIR = "attachments"
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str) -> str:
    """Reduce an uploaded name to a bare, safe basename."""
    name = _UNSAFE_NAME.sub("-", Path(str(filename)).name).strip("-.")
    if not name or name in {".", ".."}:
        raise ValueError("attachment filename is not usable")
    return name[:120]


def attachment_dir(slug: str, node_id: str, root: str | Path | None = None) -> Path:
    """Folder holding the files copied in for one node."""
    return resolve_root(root) / ATTACHMENTS_DIR / safe_filename(slug) / safe_filename(node_id)


def store_attachment(
    slug: str,
    node_id: str,
    filename: str,
    data: bytes,
    root: str | Path | None = None,
) -> Path:
    """Copy bytes into the docs folder, without overwriting an existing file."""
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"attachment is larger than {MAX_ATTACHMENT_BYTES // (1024 * 1024)}MB")
    folder = attachment_dir(slug, node_id, root)
    folder.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    target = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 2
    while target.exists():
        target = folder / f"{stem}-{counter}{suffix}"
        counter += 1
    target.write_bytes(data)
    return target


def attachment_path(
    slug: str,
    node_id: str,
    filename: str,
    root: str | Path | None = None,
) -> Path | None:
    """Resolve a stored attachment, or ``None`` if it is not in this project's folder."""
    folder = attachment_dir(slug, node_id, root)
    try:
        target = (folder / safe_filename(filename)).resolve()
        if target.is_file() and target.parent == folder.resolve():
            return target
    except (OSError, ValueError):
        return None
    return None


def delete_attachment(
    slug: str,
    node_id: str,
    filename: str,
    root: str | Path | None = None,
) -> bool:
    target = attachment_path(slug, node_id, filename, root)
    if target is None:
        return False
    target.unlink()
    return True
