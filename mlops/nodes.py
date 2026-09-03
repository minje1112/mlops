"""The node graph: every piece of work is a node that names where it came from.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Iterable, Sequence

NODE_TYPES = ("action", "note", "dataset", "experiment", "artifact")
EXPERIMENT_MODES = ("evaluation", "training", "finetune")
EXPERIMENT_STATUSES = ("running", "completed", "failed")
DATASET_SPLITS = ("raw", "train", "validation", "test", "features", "other")
ARTIFACT_KINDS = ("model", "export", "report", "deployment")
ATTACHMENT_TYPES = ("image", "pdf", "file", "link", "path")

ID_PREFIXES = {
    "action": "act",
    "note": "note",
    "dataset": "ds",
    "experiment": "exp",
    "artifact": "art",
}

# Filesystem paths written into free text, so notes can offer to open them.
PATH_PATTERN = re.compile(
    r"(?<![\w/~.-])("
    r"~/[^\s,;'\"()<>]+"          # ~/Documents/Minje
    r"|\.{1,2}/[^\s,;'\"()<>]+"   # ./data or ../shared
    r"|/(?!/)[^\s,;'\"()<>]{2,}"  # /home/rookie/Documents, but never a URL's //host
    r"|[A-Za-z]:\\[^\s,;'\"<>]+"  # C:\Users\rookie
    r")"
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_node_id(node_type: str, nodes: Sequence[dict[str, Any]]) -> str:
    """Readable, type-prefixed ids: act-001, exp-002, ds-003."""
    prefix = ID_PREFIXES[node_type]
    taken = {node.get("id") for node in nodes}
    index = sum(1 for node in nodes if node.get("type") == node_type) + 1
    while f"{prefix}-{index:03d}" in taken:
        index += 1
    return f"{prefix}-{index:03d}"


def as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def find(nodes: Sequence[dict[str, Any]], node_id: str) -> dict[str, Any] | None:
    return next((node for node in nodes if node.get("id") == node_id), None)


def require(nodes: Sequence[dict[str, Any]], node_id: str) -> dict[str, Any]:
    node = find(nodes, node_id)
    if node is None:
        raise KeyError(f"node '{node_id}' not found")
    return node


def ancestors(nodes: Sequence[dict[str, Any]], node_id: str) -> set[str]:
    """Every node this one descends from — its origin."""
    by_id = {node["id"]: node for node in nodes}
    seen: set[str] = set()
    queue = list(by_id.get(node_id, {}).get("parents", []))
    while queue:
        current = queue.pop()
        if current in seen or current not in by_id:
            continue
        seen.add(current)
        queue.extend(by_id[current].get("parents", []))
    return seen


def descendants(nodes: Sequence[dict[str, Any]], node_id: str) -> set[str]:
    """Every node that grew out of this one."""
    children: dict[str, list[str]] = {node["id"]: [] for node in nodes}
    for node in nodes:
        for parent in node.get("parents", []):
            if parent in children:
                children[parent].append(node["id"])
    seen: set[str] = set()
    queue = list(children.get(node_id, []))
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(children.get(current, []))
    return seen


def depths(nodes: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Longest path from a root, so the tree draws one row per generation."""
    by_id = {node["id"]: node for node in nodes}
    resolved: dict[str, int] = {}

    def depth_of(node_id: str, seen: frozenset[str] = frozenset()) -> int:
        if node_id in resolved:
            return resolved[node_id]
        if node_id in seen or node_id not in by_id:
            return 0
        parents = [p for p in by_id[node_id].get("parents", []) if p in by_id]
        value = max((depth_of(p, seen | {node_id}) + 1 for p in parents), default=0)
        resolved[node_id] = value
        return value

    return {node["id"]: depth_of(node["id"]) for node in nodes}


def validate_parents(
    nodes: Sequence[dict[str, Any]],
    parents: Sequence[str],
    *,
    node_id: str | None = None,
) -> list[str]:
    """Check the parent rule, de-duplicating while preserving order.

    The first node in a project is the root and takes no parents; every later
    node needs at least one that already exists. ``node_id`` is passed when
    re-parenting an existing node, so the change can be checked for cycles.
    """
    unique: list[str] = []
    for parent in parents:
        if parent not in unique:
            unique.append(parent)

    if not nodes:
        if unique:
            raise ValueError("the first node in a project is the root and cannot have parents")
        return []

    current = find(nodes, node_id) if node_id is not None else None
    if node_id is not None and current is None:
        raise KeyError(f"node '{node_id}' not found")

    if not unique:
        # The root has no parents by definition, so editing it keeps it parentless.
        if current is not None and not current.get("parents"):
            return []
        raise ValueError(
            "every node after the root needs at least one parent — "
            "name the node(s) this work grew out of"
        )

    for parent in unique:
        if parent == node_id:
            raise ValueError("a node cannot be its own parent")
        if find(nodes, parent) is None:
            raise KeyError(f"parent node '{parent}' not found")

    if node_id is not None:
        downstream = descendants(nodes, node_id)
        cycle = [parent for parent in unique if parent in downstream]
        if cycle:
            raise ValueError(f"'{cycle[0]}' descends from '{node_id}', so it cannot become its parent")

    return unique


def new_node(
    node_type: str,
    title: str,
    nodes: Sequence[dict[str, Any]],
    parents: Sequence[str] = (),
    summary: str = "",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if node_type not in NODE_TYPES:
        raise ValueError(f"node type must be one of {', '.join(NODE_TYPES)}")
    if not title.strip():
        raise ValueError("node title must not be empty")
    now = timestamp()
    return {
        "id": next_node_id(node_type, nodes),
        "type": node_type,
        "title": title.strip(),
        "summary": summary,
        "parents": validate_parents(nodes, parents),
        "created_at": now,
        "updated_at": now,
        "attachments": [],
        "data": dict(data or {}),
    }


def was_edited(node: dict[str, Any]) -> bool:
    return bool(node.get("updated_at")) and node["updated_at"] != node.get("created_at")


def next_attachment_id(node: dict[str, Any]) -> str:
    attachments = node.get("attachments", [])
    taken = {item.get("id") for item in attachments}
    index = len(attachments) + 1
    while f"att-{index:03d}" in taken:
        index += 1
    return f"att-{index:03d}"


def extract_paths(*texts: str) -> list[str]:
    """Filesystem paths mentioned in free text, in order, without duplicates."""
    found: list[str] = []
    for text in texts:
        for match in PATH_PATTERN.findall(text or ""):
            cleaned = match.rstrip(".,;:)")
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found


def graph(nodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Nodes plus parent edges and depths, ready for the dashboard to lay out."""
    known = {node["id"] for node in nodes}
    edges = [
        {"from": parent, "to": node["id"]}
        for node in nodes
        for parent in node.get("parents", [])
        if parent in known
    ]
    return {"nodes": list(nodes), "edges": edges, "depths": depths(nodes)}


def counts_by_type(nodes: Iterable[dict[str, Any]]) -> dict[str, int]:
    tally = {node_type: 0 for node_type in NODE_TYPES}
    for node in nodes:
        if node.get("type") in tally:
            tally[node["type"]] += 1
    return tally
