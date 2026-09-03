import base64
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlops import MLOpsWriter, list_projects, load_project, save_project
from mlops.store import ROOT_ENV_VAR, default_root, get_project, project_slug
from mlops.web import DashboardHandler, metric_summary, project_detail


@pytest.fixture
def docs_root(tmp_path):
    root = tmp_path / "docs"

    churn = MLOpsWriter(project_name="Churn Prediction", objective="improve recall")
    churn.add_action("ingest data", "loaded source table")
    churn.set_dataset("train", "s3://bucket/churn/v1/train.parquet", note="raw export")
    churn.log_dataset_change("train", "filtering", "Removed null target rows")
    churn.add_training_run("xgboost", "train", metrics={"recall": 0.82}, hyperparameters={"max_depth": 6})
    churn.add_training_run("lightgbm", "train", metrics={"recall": 0.88})
    churn.suggest_utilities()
    churn.save(root)

    fraud = MLOpsWriter(project_name="fraud-detection")
    fraud.set_dataset("train", "data/fraud.csv")
    save_project(fraud, root)

    (root / "not-a-project.json").write_text("{ broken", encoding="utf-8")
    return root


def serve(root, allow_open=True, host="127.0.0.1"):
    handler = partial(DashboardHandler, root=root, allow_open=allow_open)
    httpd = ThreadingHTTPServer((host, 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://{host}:{httpd.server_address[1]}"


def call(base, path, method, payload=None, content_type="application/json", origin=None):
    """Issue a write request and return (status, decoded body)."""
    headers = {}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = content_type
    if origin:
        headers["Origin"] = origin
    request = Request(f"{base}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_saving_creates_slugged_files_and_updates_in_place(tmp_path):
    root = tmp_path / "docs"
    writer = MLOpsWriter(project_name="Churn Prediction")
    first = writer.save(root)
    assert first.name == "churn-prediction.json"

    writer.add_action("retrain", "weekly refresh")
    assert writer.save(root) == first
    assert len(list(root.glob("*.json"))) == 1
    assert len(json.loads(first.read_text())["nodes"]) == 1


def test_distinct_projects_with_the_same_slug_do_not_overwrite(tmp_path):
    root = tmp_path / "docs"
    MLOpsWriter(project_name="churn prediction").save(root)
    MLOpsWriter(project_name="churn/prediction").save(root)
    assert sorted(path.name for path in root.glob("*.json")) == [
        "churn-prediction-2.json",
        "churn-prediction.json",
    ]


def test_project_slug_falls_back_for_unusable_names():
    assert project_slug("!!!") == "project"


def test_listing_skips_unreadable_files_and_sorts_by_activity(docs_root):
    projects = list_projects(docs_root)
    assert [project["slug"] for project in projects] == ["fraud-detection", "churn-prediction"]

    churn = projects[1]
    assert churn["project_name"] == "Churn Prediction"
    assert churn["counts"] == {
        "action": 1, "note": 0, "dataset": 1, "experiment": 2, "artifact": 0, "nodes": 4
    }
    assert churn["last_activity"] >= churn["created_at"]


def test_listing_missing_root_is_empty(tmp_path):
    assert list_projects(tmp_path / "nope") == []


def test_saved_project_reloads_into_a_writer(docs_root):
    writer = load_project("churn-prediction", docs_root)
    assert isinstance(writer, MLOpsWriter)
    assert writer.project_name == "Churn Prediction"

    writer.add_training_run("catboost", "train", metrics={"recall": 0.9})
    writer.save(docs_root)
    assert len(load_project("churn-prediction", docs_root).experiments) == 3

    with pytest.raises(KeyError):
        load_project("missing", docs_root)


def test_metric_summary_aggregates_each_metric():
    experiments = [
        {"data": {"metrics": {"recall": 0.8, "auc": 0.7}}},
        {"data": {"metrics": {"recall": 0.9}}},
    ]
    summary = metric_summary(experiments)
    assert list(summary) == ["recall", "auc"]
    assert summary["recall"] == pytest.approx({"count": 2.0, "best": 0.9, "average": 0.85})
    assert summary["auc"] == pytest.approx({"count": 1.0, "best": 0.7, "average": 0.7})
    assert metric_summary([]) == {}


def test_project_detail_bundles_summary_graph_and_metrics(docs_root):
    detail = project_detail("churn-prediction", docs_root)
    assert detail["summary"]["project_name"] == "Churn Prediction"
    assert detail["metric_summary"]["recall"]["best"] == 0.88
    assert [node["id"] for node in detail["graph"]["nodes"]] == ["act-001", "ds-001", "exp-001", "exp-002"]
    assert detail["graph"]["depths"]["exp-001"] == 2
    assert project_detail("missing", docs_root) is None


def test_http_routes_serve_dashboard_and_project_data(docs_root):
    httpd, base = serve(docs_root)
    try:
        with urlopen(f"{base}/") as response:
            page = response.read().decode()
        assert response.headers["Content-Type"].startswith("text/html")
        assert "MLOps Tracker" in page

        with urlopen(f"{base}/api/projects") as response:
            index = json.load(response)
        assert {project["slug"] for project in index["projects"]} == {"churn-prediction", "fraud-detection"}
        assert index["root"] == str(docs_root.resolve())
        assert index["can_open_paths"] is True

        with urlopen(f"{base}/api/projects/churn-prediction") as response:
            detail = json.load(response)
        assert detail["document"]["objective"] == "improve recall"
        assert detail["metric_summary"]["recall"]["count"] == 2.0
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.parametrize("path", ["/api/projects/missing", "/api/projects/../pyproject", "/nope"])
def test_unknown_routes_return_404_json(docs_root, path):
    httpd, base = serve(docs_root)
    try:
        with pytest.raises(HTTPError) as excinfo:
            urlopen(f"{base}{path}")
        assert excinfo.value.code == 404
        assert "error" in json.loads(excinfo.value.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_creating_a_project_over_http(docs_root):
    httpd, base = serve(docs_root)
    try:
        status, created = call(base, "/api/projects", "POST", {"project_name": "New Ranker", "objective": "ndcg"})
        assert (status, created["slug"]) == (200, "new-ranker")
        assert get_project("new-ranker", docs_root)["objective"] == "ndcg"

        assert call(base, "/api/projects", "POST", {"project_name": "New Ranker"})[0] == 409
        assert call(base, "/api/projects", "POST", {"project_name": "  "})[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_adding_and_editing_actions_over_http(docs_root):
    httpd, base = serve(docs_root)
    try:
        status, added = call(
            base, "/api/projects/fraud-detection/actions", "POST",
            {"title": "retrain", "details": "weekly refresh", "category": "modeling"},
        )
        assert status == 200
        action_id = added["action"]["id"]
        assert added["action"]["updated_at"] == added["action"]["created_at"]

        status, edited = call(
            base, f"/api/projects/fraud-detection/actions/{action_id}", "PATCH",
            {"details": "weekly refresh, now nightly"},
        )
        assert status == 200
        assert edited["action"]["title"] == "retrain"
        assert edited["action"]["summary"] == "weekly refresh, now nightly"
        assert edited["action"]["updated_at"] > edited["action"]["created_at"]

        stored = get_project("fraud-detection", docs_root)["nodes"][-1]
        assert stored["summary"] == "weekly refresh, now nightly"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------- nodes

def test_building_a_tree_over_http(docs_root):
    httpd, base = serve(docs_root)
    try:
        status, baseline = call(base, "/api/projects/churn-prediction/nodes", "POST", {
            "type": "experiment", "title": "baseline eval", "parents": ["act-001"],
            "summary": "ran the stock model", "data": {"mode": "evaluation", "metrics": {"recall": 0.4}},
        })
        assert status == 200
        assert baseline["node"]["parents"] == ["act-001"]
        assert baseline["node"]["data"]["mode"] == "evaluation"

        status, note = call(base, "/api/projects/churn-prediction/nodes", "POST", {
            "type": "note", "title": "weak on rare classes",
            "parents": [baseline["node"]["id"]], "summary": "see /tmp",
        })
        assert status == 200

        status, dataset = call(base, "/api/projects/churn-prediction/nodes", "POST", {
            "type": "dataset", "title": "train", "parents": [note["node"]["id"]],
            "data": {"data_ref": "s3://bucket/churn/v2/train.parquet", "split": "train"},
        })
        assert (status, dataset["node"]["data"]["version"]) == (200, 2)

        status, edited = call(
            base, f"/api/projects/churn-prediction/nodes/{note['node']['id']}", "PATCH",
            {"summary": "weak on rare classes, see /tmp"},
        )
        assert status == 200
        assert edited["node"]["updated_at"] > edited["node"]["created_at"]

        with urlopen(f"{base}/api/projects/churn-prediction") as response:
            detail = json.load(response)
        assert detail["paths"][note["node"]["id"]] == ["/tmp"]
        # v2 hangs off both the note it came from and the version before it.
        assert set(dataset["node"]["parents"]) == {"ds-001", note["node"]["id"]}
        assert detail["graph"]["depths"][dataset["node"]["id"]] == 3
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_editing_the_root_node_over_http(docs_root):
    httpd, base = serve(docs_root)
    try:
        status, edited = call(base, "/api/projects/churn-prediction/nodes/act-001", "PATCH",
                              {"title": "ingest data", "summary": "loaded the source table", "parents": []})
        assert (status, edited["node"]["parents"]) == (200, [])
        assert edited["node"]["summary"] == "loaded the source table"
        assert get_project("churn-prediction", docs_root)["nodes"][0]["parents"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_editing_an_experiment_keeps_what_it_did_not_touch(docs_root):
    """The dashboard's Edit form posts every field; the machine-recorded ones must survive."""
    writer = load_project("churn-prediction", docs_root)
    try:
        with writer.run("granite", "train", mode="finetune", parameters={"lr": 2e-05, "bf16": True}) as run:
            run.log_params(adapter=None, epochs=10)
            raise RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")
    except RuntimeError:
        pass
    writer.save(docs_root)
    node_id = writer.experiments[-1]["id"]

    httpd, base = serve(docs_root)
    try:
        status, edited = call(base, f"/api/projects/churn-prediction/nodes/{node_id}", "PATCH", {
            "title": "granite", "summary": "bf16 GEMM died, retry with --fp16",
            "data": {"model_name": "granite", "mode": "finetune", "status": "failed",
                     "architecture": "LoRA r=32",
                     "metrics": {},
                     "parameters": {"lr": 2e-05, "bf16": True, "adapter": None, "epochs": 10}},
        })
        assert status == 200
        data = edited["node"]["data"]
        assert edited["node"]["summary"] == "bf16 GEMM died, retry with --fp16"
        assert data["parameters"] == {"lr": 2e-05, "bf16": True, "adapter": None, "epochs": 10}
        # untouched machine-recorded fields
        assert data["error"].startswith("RuntimeError: CUBLAS")
        assert data["environment"]["python"]
        assert data["duration_seconds"] >= 0
        assert data["dataset_roles"]

        # status is editable, and validated
        assert call(base, f"/api/projects/churn-prediction/nodes/{node_id}", "PATCH",
                    {"data": {"status": "completed"}})[0] == 200
        code, body = call(base, f"/api/projects/churn-prediction/nodes/{node_id}", "PATCH",
                          {"data": {"status": "exploded"}})
        assert (code, "experiment status" in body["error"]) == (400, True)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_node_endpoints_enforce_the_graph_rules(docs_root):
    httpd, base = serve(docs_root)
    try:
        # An empty parent list is refused for every type, never re-homed on the root.
        for node_type, data in [
            ("note", {}), ("action", {}),
            ("experiment", {"mode": "evaluation"}),
            ("artifact", {"artifact_ref": "s3://b/m.pt"}),
        ]:
            status, body = call(base, "/api/projects/churn-prediction/nodes", "POST",
                                {"type": node_type, "title": "orphan", "parents": [], "data": data})
            assert (node_type, status, "at least one parent" in body["error"]) == (node_type, 400, True)
        assert len(get_project("churn-prediction", docs_root)["nodes"]) == 4

        assert call(base, "/api/projects/churn-prediction/nodes", "POST",
                    {"type": "note", "title": "x", "parents": ["nope"]})[0] == 404
        assert call(base, "/api/projects/churn-prediction/nodes", "POST",
                    {"type": "note", "title": "", "parents": ["act-001"]})[0] == 400
        assert call(base, "/api/projects/churn-prediction/nodes", "POST",
                    {"type": "banana", "title": "x", "parents": ["act-001"]})[0] == 400
        assert call(base, "/api/projects/churn-prediction/nodes", "POST",
                    {"type": "note", "title": "x", "parents": "act-001"})[0] == 400
        assert call(base, "/api/projects/churn-prediction/nodes/nope", "PATCH", {"title": "x"})[0] == 404
        assert call(base, "/api/projects/churn-prediction/nodes/act-001", "PATCH", {})[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_write_endpoints_validate_their_input(docs_root):
    httpd, base = serve(docs_root)
    try:
        assert call(base, "/api/projects/fraud-detection/actions", "POST", {"details": "no title"})[0] == 400
        assert call(base, "/api/projects/missing/actions", "POST", {"title": "x"})[0] == 404
        assert call(base, "/api/projects/churn-prediction/actions/act-404", "PATCH", {"title": "x"})[0] == 404
        assert call(base, "/api/projects/churn-prediction/actions/act-001", "PATCH", {})[0] == 400
        assert call(base, "/api/projects", "POST", {"project_name": "x"}, content_type="text/plain")[0] == 415
        assert call(base, "/api/projects", "POST", {"project_name": "x"}, origin="http://evil.test")[0] == 403
        assert call(base, "/api/nope", "POST", {})[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


# ----------------------------------------------------------------- attachments

def test_attachments_round_trip_through_the_server(docs_root):
    httpd, base = serve(docs_root)
    try:
        content = b"\x89PNG not really a png"
        status, uploaded = call(base, "/api/projects/churn-prediction/nodes/act-001/attachments", "POST", {
            "type": "upload", "filename": "confusion matrix.png",
            "content_base64": base64.b64encode(content).decode(), "label": "confusion matrix",
        })
        assert (status, uploaded["attachment"]["type"]) == (200, "image")
        filename = uploaded["attachment"]["filename"]
        assert filename == "confusion-matrix.png"

        with urlopen(f"{base}/api/attachments/churn-prediction/act-001/{filename}") as response:
            assert response.read() == content
            assert response.headers["Content-Type"] == "image/png"
            assert "inline" in response.headers["Content-Disposition"]

        status, link = call(base, "/api/projects/churn-prediction/nodes/act-001/attachments", "POST",
                            {"type": "link", "href": "https://example.org/card", "label": "card"})
        assert (status, link["attachment"]["type"]) == (200, "link")
        status, path = call(base, "/api/projects/churn-prediction/nodes/act-001/attachments", "POST",
                            {"type": "path", "href": str(docs_root)})
        assert (status, path["attachment"]["type"]) == (200, "path")

        stored = get_project("churn-prediction", docs_root)["nodes"][0]["attachments"]
        assert [item["type"] for item in stored] == ["image", "link", "path"]

        status, _ = call(base, f"/api/projects/churn-prediction/nodes/act-001/attachments/{uploaded['attachment']['id']}", "DELETE")
        assert status == 200
        assert not (docs_root / "attachments" / "churn-prediction" / "act-001" / filename).exists()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_attachment_endpoints_reject_bad_input(docs_root):
    httpd, base = serve(docs_root)
    try:
        node_url = "/api/projects/churn-prediction/nodes/act-001/attachments"
        assert call(base, node_url, "POST", {"type": "upload", "content_base64": "aGk="})[0] == 400
        assert call(base, node_url, "POST", {"type": "upload", "filename": "a.png", "content_base64": "!!"})[0] == 400
        assert call(base, node_url, "POST", {"type": "upload", "filename": "a.png", "content_base64": ""})[0] == 400
        assert call(base, node_url, "POST", {"type": "link", "href": "not-a-url"})[0] == 400
        assert call(base, node_url, "POST", {"type": "shrug", "href": "x"})[0] == 400
        assert call(base, "/api/projects/churn-prediction/nodes/nope/attachments", "POST",
                    {"type": "link", "href": "https://x.test"})[0] == 404
        assert call(base, f"{node_url}/att-404", "DELETE")[0] == 404

        with pytest.raises(HTTPError) as excinfo:
            urlopen(f"{base}/api/attachments/churn-prediction/act-001/missing.png")
        assert excinfo.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_stored_attachments_cannot_escape_the_project_folder(docs_root, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    httpd, base = serve(docs_root)
    try:
        for target in ["../../secret.txt", "..%2f..%2fsecret.txt", "%2e%2e/secret.txt"]:
            with pytest.raises(HTTPError) as excinfo:
                urlopen(f"{base}/api/attachments/churn-prediction/act-001/{target}")
            assert excinfo.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


# -------------------------------------------------------------- opening folders

def test_opening_a_folder_calls_the_file_manager(docs_root, monkeypatch):
    launched = []
    monkeypatch.setattr("mlops.opener._launch", lambda argv: launched.append(argv))

    httpd, base = serve(docs_root)
    try:
        status, body = call(base, "/api/open", "POST", {"path": str(docs_root)})
        assert status == 200
        assert body["opened"] == str(docs_root.resolve())
        assert launched and str(docs_root.resolve()) in launched[0]

        # A file is revealed inside its folder rather than opened.
        target = docs_root / "churn-prediction.json"
        status, body = call(base, "/api/open", "POST", {"path": str(target)})
        assert (status, body["opened"]) == (200, str(docs_root.resolve()))

        assert call(base, "/api/open", "POST", {"path": str(docs_root / "nope")})[0] == 404
        assert call(base, "/api/open", "POST", {"path": ""})[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_opening_folders_is_refused_when_disabled(docs_root, monkeypatch):
    monkeypatch.setattr("mlops.opener._launch", lambda argv: pytest.fail("must not launch"))
    httpd, base = serve(docs_root, allow_open=False)
    try:
        status, body = call(base, "/api/open", "POST", {"path": str(docs_root)})
        assert (status, "disabled" in body["error"]) == (403, True)
        with urlopen(f"{base}/api/projects") as response:
            assert json.load(response)["can_open_paths"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


# ------------------------------------------------------------------ docs root

def test_root_defaults_to_cwd_then_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv(ROOT_ENV_VAR, raising=False)
    assert default_root() == Path("mlops_docs")

    shared = tmp_path / "shared"
    monkeypatch.setenv(ROOT_ENV_VAR, str(shared))
    assert default_root() == shared

    # Saving from any working directory lands in the shared folder.
    monkeypatch.chdir(tmp_path)
    saved = MLOpsWriter(project_name="cross-repo").save()
    assert saved == shared / "cross-repo.json"
    assert [project["slug"] for project in list_projects()] == ["cross-repo"]
    assert load_project("cross-repo").project_name == "cross-repo"


def test_explicit_root_overrides_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "shared"))
    explicit = tmp_path / "explicit"
    saved = MLOpsWriter(project_name="pinned").save(explicit)
    assert saved.parent == explicit
    assert list_projects(explicit)[0]["slug"] == "pinned"
    assert list_projects() == []
