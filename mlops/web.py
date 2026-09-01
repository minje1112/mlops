"""Local web dashboard for browsing and updating MLOps project documents.

Standard library only: run ``python3 -m mlops.web`` (or ``mlops-web``) and open
the printed URL. Documents are the JSON files written by
:meth:`mlops.MLOpsWriter.save`.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import ipaddress
import json
import mimetypes
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .nodes import extract_paths
from .opener import OpenError, open_in_file_manager
from .store import (
    MAX_ATTACHMENT_BYTES,
    ROOT_ENV_VAR,
    attachment_path,
    create_project,
    get_project,
    list_projects,
    load_project,
    project_slug,
    resolve_root,
    store_attachment,
    summarize_project,
)
from .utils import summarize_metrics

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"
MAX_BODY_BYTES = 256 * 1024
# Uploads are base64 in a JSON body, which costs about a third on top.
MAX_UPLOAD_BODY_BYTES = MAX_ATTACHMENT_BYTES * 4 // 3 + 64 * 1024

# One writer at a time: the dashboard is threaded and mutations rewrite whole files.
_WRITE_LOCK = threading.Lock()


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def metric_summary(experiments: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """count/best/average per metric name seen across a project's experiments."""
    runs = [node["data"] for node in experiments]
    names: list[str] = []
    for run in runs:
        for name in run.get("metrics", {}):
            if name not in names:
                names.append(name)
    return {name: summarize_metrics(runs, name) for name in names}


def project_detail(slug: str, root: str | Path | None = None) -> dict[str, Any] | None:
    document = get_project(slug, root)
    if document is None:
        return None
    writer = load_project(slug, root)
    return {
        "summary": summarize_project(slug, writer.as_dict()),
        "document": writer.as_dict(),
        "graph": writer.graph(),
        "metric_summary": metric_summary(writer.experiments),
        "paths": {
            node["id"]: extract_paths(node.get("summary", ""), node.get("title", ""))
            for node in writer.nodes
        },
    }


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost", ""}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "mlops-web"
    protocol_version = "HTTP/1.1"

    def __init__(
        self,
        *args: Any,
        root: str | Path | None = None,
        allow_open: bool = True,
        **kwargs: Any,
    ) -> None:
        self.root = resolve_root(root)
        self.allow_open = allow_open
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ routing

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self._path()
        if path == "/":
            self._send_html()
        elif path == "/api/projects":
            self._send_json({
                "root": str(self.root.resolve()),
                "projects": list_projects(self.root),
                "can_open_paths": self._can_open(),
            })
        elif path.startswith("/api/attachments/"):
            self._send_attachment(path[len("/api/attachments/"):])
        elif path.startswith("/api/projects/"):
            slug = path[len("/api/projects/"):]
            detail = project_detail(slug, self.root) if "/" not in slug else None
            if detail is None:
                self._send_json({"error": f"project '{slug}' not found"}, status=404)
            else:
                self._send_json(detail)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self._mutate(self._route_post, large=self._path().endswith("/attachments"))

    def do_PATCH(self) -> None:  # noqa: N802 - http.server API
        self._mutate(self._route_patch)

    def do_DELETE(self) -> None:  # noqa: N802 - http.server API
        self._mutate(self._route_delete, body=False)

    def _segments(self, path: str) -> list[str]:
        if not path.startswith("/api/projects/"):
            return []
        return path[len("/api/projects/"):].split("/")

    def _route_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/api/projects":
            return self._create_project(payload)
        if path == "/api/open":
            return self._open_path(payload)
        parts = self._segments(path)
        if len(parts) == 2 and parts[1] == "actions":
            return self._add_action(parts[0], payload)
        if len(parts) == 2 and parts[1] == "nodes":
            return self._add_node(parts[0], payload)
        if len(parts) == 4 and parts[1] == "nodes" and parts[3] == "attachments":
            return self._add_attachment(parts[0], parts[2], payload)
        raise ApiError("not found", status=404)

    def _route_patch(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        parts = self._segments(path)
        if len(parts) == 3 and parts[1] == "actions":
            return self._edit_action(parts[0], parts[2], payload)
        if len(parts) == 3 and parts[1] == "nodes":
            return self._edit_node(parts[0], parts[2], payload)
        raise ApiError("not found", status=404)

    def _route_delete(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        parts = self._segments(path)
        if len(parts) == 5 and parts[1] == "nodes" and parts[3] == "attachments":
            return self._remove_attachment(parts[0], parts[2], parts[4])
        raise ApiError("not found", status=404)

    # ----------------------------------------------------------------- handlers

    def _create_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("project_name", "")).strip()
        if not name:
            raise ApiError("project_name is required")
        try:
            writer = create_project(name, str(payload.get("objective", "")), self.root)
        except FileExistsError as error:
            raise ApiError(str(error), status=409) from error
        except ValueError as error:
            raise ApiError(str(error)) from error
        return {"slug": project_slug(writer.project_name), "project_name": writer.project_name}

    def _add_action(self, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
        writer = self._load(slug)
        title = str(payload.get("title", "")).strip()
        if not title:
            raise ApiError("title is required")
        with self._reporting():
            action = writer.add_action(
                title,
                str(payload.get("details", "")).strip(),
                str(payload.get("category", "operation")).strip() or "operation",
                parents=payload.get("parents"),
            )
        writer.save(self.root)
        return {"action": action, "node": action}

    def _edit_action(self, slug: str, action_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        writer = self._load(slug)
        fields = {key: payload[key] for key in ("title", "details", "category") if key in payload}
        if "title" in fields and not str(fields["title"]).strip():
            raise ApiError("title must not be empty")
        with self._reporting():
            action = writer.edit_action(action_id, **{key: str(value) for key, value in fields.items()})
        writer.save(self.root)
        return {"action": action, "node": action}

    def _add_node(self, slug: str, payload: dict[str, Any]) -> dict[str, Any]:
        writer = self._load(slug)
        node_type = str(payload.get("type", "")).strip()
        title = str(payload.get("title", "")).strip()
        if not title:
            raise ApiError("title is required")
        parents = payload.get("parents") or []
        if not isinstance(parents, list):
            raise ApiError("parents must be a list of node ids")
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise ApiError("data must be an object")

        # ``parents`` is passed through as given: an empty list must be refused
        # rather than quietly re-homed on the root.
        with self._reporting():
            if node_type == "dataset":
                node = writer.add_dataset(
                    title, str(data.get("data_ref", "")), split=str(data.get("split", "train")),
                    note=str(payload.get("summary", "")), parents=parents,
                )
            elif node_type == "experiment":
                node = writer.add_experiment(
                    str(data.get("model_name") or title), parents=parents,
                    mode=str(data.get("mode", "evaluation")),
                    datasets=data.get("dataset_roles") or None,
                    metrics=data.get("metrics") or {},
                    parameters=data.get("parameters") or {},
                    architecture=str(data.get("architecture", "")),
                    notes=str(payload.get("summary", "")), title=title,
                )
            elif node_type == "artifact":
                node = writer.add_artifact(
                    title, str(data.get("artifact_ref", "")), kind=str(data.get("kind", "model")),
                    note=str(payload.get("summary", "")), parents=parents,
                )
            else:
                node = writer.add_node(
                    node_type, title, parents, summary=str(payload.get("summary", "")),
                    **({"category": str(data.get("category", "operation"))} if node_type == "action" else {}),
                )
        writer.save(self.root)
        return {"node": node}

    def _edit_node(self, slug: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        writer = self._load(slug)
        fields: dict[str, Any] = {}
        if "title" in payload:
            fields["title"] = str(payload["title"])
        if "summary" in payload:
            fields["summary"] = str(payload["summary"])
        if "parents" in payload:
            if not isinstance(payload["parents"], list):
                raise ApiError("parents must be a list of node ids")
            fields["parents"] = payload["parents"]
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise ApiError("data must be an object")
        if not fields and not data:
            raise ApiError("nothing to change")
        with self._reporting():
            node = writer.edit_node(node_id, **fields, **data)
        writer.save(self.root)
        return {"node": node}

    def _add_attachment(self, slug: str, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        writer = self._load(slug)
        kind = str(payload.get("type", "")).strip()
        label = str(payload.get("label", ""))
        with self._reporting():
            node = writer.get_node(node_id)
            if kind == "link":
                attachment = writer.attach_link(node, str(payload.get("href", "")), label)
            elif kind == "path":
                attachment = writer.attach_path(node, str(payload.get("href", "")), label)
            elif kind == "upload":
                attachment = self._store_upload(writer, slug, node, payload, label)
            else:
                raise ApiError("attachment type must be 'link', 'path' or 'upload'")
        writer.save(self.root)
        return {"attachment": attachment, "node": node}

    def _store_upload(
        self,
        writer: Any,
        slug: str,
        node: dict[str, Any],
        payload: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        filename = str(payload.get("filename", "")).strip()
        if not filename:
            raise ApiError("filename is required for an upload")
        try:
            content = base64.b64decode(str(payload.get("content_base64", "")), validate=True)
        except (binascii.Error, ValueError):
            raise ApiError("content_base64 is not valid base64") from None
        if not content:
            raise ApiError("uploaded file is empty")
        try:
            stored = store_attachment(slug, node["id"], filename, content, self.root)
        except ValueError as error:
            raise ApiError(str(error), status=413) from error
        return writer.attach_stored(node, stored.name, len(content), label or filename)

    def _remove_attachment(self, slug: str, node_id: str, attachment_id: str) -> dict[str, Any]:
        writer = self._load(slug)
        with self._reporting():
            removed = writer.remove_attachment(node_id, attachment_id)
        writer.save(self.root)
        return {"removed": removed}

    def _open_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._can_open():
            raise ApiError(
                "opening folders is disabled — it is only available on a localhost bind "
                "and without --no-open",
                status=403,
            )
        try:
            return open_in_file_manager(str(payload.get("path", "")))
        except OpenError as error:
            raise ApiError(str(error), status=404) from error

    def _can_open(self) -> bool:
        host = self.server.server_address[0] if self.server else ""
        return bool(self.allow_open) and is_loopback(str(host))

    def _load(self, slug: str) -> Any:
        try:
            return load_project(slug, self.root)
        except KeyError as error:
            raise ApiError(f"project '{slug}' not found", status=404) from error

    # ------------------------------------------------------------------ plumbing

    class _Reporting:
        """Turn writer-level errors into API errors with the right status."""

        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            if exc_type is None:
                return False
            if isinstance(exc, KeyError):
                raise ApiError(str(exc).strip("\"'"), status=404) from exc
            if isinstance(exc, (ValueError, FileNotFoundError)):
                raise ApiError(str(exc)) from exc
            return False

    def _reporting(self) -> "DashboardHandler._Reporting":
        return self._Reporting()

    def _path(self) -> str:
        return unquote(urlparse(self.path).path).rstrip("/") or "/"

    def _mutate(self, route: Callable[..., dict[str, Any]], body: bool = True, large: bool = False) -> None:
        try:
            self._check_origin()
            payload = self._read_json(large=large) if body else {}
            with _WRITE_LOCK:
                result = route(self._path(), payload)
        except ApiError as error:
            self._send_json({"error": error.message}, status=error.status)
        else:
            self._send_json(result, status=200)

    def _check_origin(self) -> None:
        """Reject cross-site writes; a browser sends Origin on every POST/PATCH."""
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            raise ApiError("cross-origin requests are not allowed", status=403)

    def _read_json(self, large: bool = False) -> dict[str, Any]:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if content_type != "application/json":
            raise ApiError("Content-Type must be application/json", status=415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError("invalid Content-Length") from None
        if length > (MAX_UPLOAD_BODY_BYTES if large else MAX_BODY_BYTES):
            raise ApiError("request body too large", status=413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError("body must be valid JSON") from None
        if not isinstance(payload, dict):
            raise ApiError("body must be a JSON object")
        return payload

    def _send_attachment(self, tail: str) -> None:
        parts = tail.split("/")
        if len(parts) != 3:
            self._send_json({"error": "not found"}, status=404)
            return
        slug, node_id, filename = parts
        target = attachment_path(slug, node_id, filename, self.root)
        if target is None:
            self._send_json({"error": "attachment not found"}, status=404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._respond(target.read_bytes(), content_type, inline=target.name)

    def _send_html(self) -> None:
        try:
            body = DASHBOARD.read_bytes()
        except OSError:
            self._send_json({"error": "dashboard template missing"}, status=500)
            return
        self._respond(body, "text/html; charset=utf-8")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._respond(body, "application/json; charset=utf-8", status=status)

    def _respond(self, body: bytes, content_type: str, status: int = 200, inline: str = "") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if inline:
            self.send_header("Content-Disposition", f'inline; filename="{inline}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        return  # quiet by default; the dashboard polls on refresh


def serve(
    root: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    allow_open: bool = True,
) -> None:
    """Serve the dashboard until interrupted."""
    root = resolve_root(root)
    handler = partial(DashboardHandler, root=root, allow_open=allow_open)
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as error:
        raise SystemExit(f"cannot bind {host}:{port} ({error}); try --port 0 to pick a free port") from error
    resolved = root.resolve()
    print(f"mlops dashboard: http://{host}:{httpd.server_address[1]}", flush=True)
    print(f"reading projects from: {resolved}", flush=True)
    if not resolved.is_dir():
        print("(folder is created when you add your first project)", flush=True)
    if not (allow_open and is_loopback(host)):
        print("(opening folders in the file manager is disabled)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Browse your MLOps projects and trackings in a browser.")
    parser.add_argument(
        "--root",
        default=None,
        help=f"folder holding saved project JSON files (default: ${ROOT_ENV_VAR} or ./mlops_docs)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: localhost only)")
    parser.add_argument("--port", type=int, default=8787, help="port to bind (0 picks a free one)")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="disable opening folders in the desktop file manager",
    )
    args = parser.parse_args(argv)
    serve(root=args.root, host=args.host, port=args.port, allow_open=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
