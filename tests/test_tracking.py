import pytest

from mlops import MLOpsWriter, evaluate


@pytest.fixture
def writer():
    project = MLOpsWriter(project_name="lineage-demo")
    project.add_action("kickoff", "scoped the project")
    project.set_dataset("raw", "s3://bucket/raw/events.parquet")
    project.set_dataset("train", "s3://bucket/curated/train.parquet", derived_from="raw")
    return project


# --------------------------------------------------------------------- actions

def test_actions_record_creation_and_last_edited_time(writer):
    action = writer.add_action("ingest", "loaded events")
    assert action["id"] == "act-002"
    assert action["updated_at"] == action["created_at"]
    assert not MLOpsWriter.action_was_edited(action)

    edited = writer.edit_action("act-002", details="loaded 14 months of events")
    assert edited["summary"] == "loaded 14 months of events"
    assert edited["title"] == "ingest"
    assert edited["updated_at"] > edited["created_at"]
    assert MLOpsWriter.action_was_edited(edited)


def test_actions_can_be_edited_by_index_and_reject_empty_edits(writer):
    writer.add_action("second", "b")
    assert writer.edit_action(1, title="renamed")["id"] == "act-002"
    assert writer.get_action("act-001")["title"] == "kickoff"

    with pytest.raises(ValueError):
        writer.edit_action("act-001")
    with pytest.raises(KeyError):
        writer.edit_action("act-404", title="nope")
    with pytest.raises(KeyError):
        writer.edit_action(9, title="nope")


def test_legacy_documents_convert_into_the_node_graph():
    restored = MLOpsWriter.from_dict(
        {
            "project_name": "legacy",
            "created_at": "2026-01-01T00:00:00+00:00",
            "actions": [
                {"id": "act-001", "timestamp": "2026-01-01T01:00:00+00:00", "title": "old", "details": "d", "category": "operation"}
            ],
            "datasets": {
                "train": {"name": "train", "data_ref": "s3://b/train", "note": "", "changes": [], "derived_from": []}
            },
            "training_runs": [
                {"timestamp": "2026-01-01T02:00:00+00:00", "model_name": "m", "dataset_name": "train",
                 "metrics": {"f1": 0.5}, "hyperparameters": {"lr": 0.1}}
            ],
            "artifacts": [
                {"name": "m.txt", "artifact_ref": "s3://b/m.txt", "kind": "model", "run_id": "exp-001",
                 "derived_from": [], "note": "", "timestamp": "t", "updated_at": "t"}
            ],
        }
    )
    assert [(node["id"], node["type"], node["parents"]) for node in restored.nodes] == [
        ("act-001", "action", []),
        ("ds-001", "dataset", ["act-001"]),
        ("exp-001", "experiment", ["ds-001"]),
        ("art-001", "artifact", ["exp-001"]),
    ]
    run = restored.experiments[0]
    assert run["data"]["parameters"] == {"lr": 0.1}
    assert run["data"]["status"] == "completed"
    assert run["data"]["mode"] == "training"
    assert restored.dataset_versions[0]["data"]["version"] == 1


# ------------------------------------------------------- experiment  tracking

def test_run_context_captures_params_metrics_timing_and_environment(writer):
    with writer.run("lightgbm", "train", architecture="gbdt, 64 leaves") as run:
        run.log_params(learning_rate=0.05, num_leaves=64)
        run.log_params({"seed": 7})
        run.log_metrics(pr_auc=0.71)

    node = writer.experiments[0]
    payload = node["data"]
    assert node["id"] == "exp-001"
    assert node["parents"] == [writer.latest_dataset("train")["id"]]
    assert payload["status"] == "completed"
    assert payload["mode"] == "training"
    assert payload["architecture"] == "gbdt, 64 leaves"
    assert payload["parameters"] == {"learning_rate": 0.05, "num_leaves": 64, "seed": 7}
    assert payload["metrics"] == {"pr_auc": 0.71}
    assert payload["duration_seconds"] >= 0
    assert payload["environment"]["python"] and payload["environment"]["platform"]
    assert payload["dataset_roles"] == {writer.latest_dataset("train")["id"]: "train"}


def test_failed_run_is_recorded_and_error_reraised(writer):
    with pytest.raises(RuntimeError, match="cuda oom"):
        with writer.run("resnet", "train") as run:
            run.log_params(batch_size=512)
            raise RuntimeError("cuda oom")

    payload = writer.experiments[0]["data"]
    assert payload["status"] == "failed"
    assert payload["error"] == "RuntimeError: cuda oom"
    assert payload["parameters"] == {"batch_size": 512}


def test_evaluation_metrics_are_logged_automatically(writer):
    with writer.run("logreg", "train") as run:
        scores = run.log_evaluation([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])

    assert scores == {"accuracy": 0.8, "precision": 1.0, "recall": 0.666667, "f1": 0.8, "support": 5.0}
    assert writer.experiments[0]["data"]["metrics"] == scores


def test_evaluation_supports_prefixes_and_multiclass(writer):
    with writer.run("clf", "train") as run:
        run.log_evaluation(["a", "b", "c", "a"], ["a", "b", "a", "a"], prefix="val_")
    metrics = writer.experiments[0]["data"]["metrics"]
    assert metrics["val_accuracy"] == 0.75
    assert 0 < metrics["val_f1"] < 1


def test_evaluate_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [])
    with pytest.raises(ValueError, match="y_true has 2 items"):
        evaluate([1, 0], [1])
    with pytest.raises(ValueError, match="average must be"):
        evaluate([1, 0], [1, 0], average="weighted")
    with pytest.raises(ValueError, match="positive_label"):
        evaluate([1, 0], [1, 0], positive_label=9)


def test_add_training_run_still_accepts_hyperparameters(writer):
    run = writer.add_training_run(
        "xgboost", "train", metrics={"recall": 0.8}, hyperparameters={"max_depth": 6}, parameters={"seed": 1}
    )
    assert run["data"]["parameters"] == {"max_depth": 6, "seed": 1}
    assert run["data"]["status"] == "completed"
    assert run["id"] == "exp-001"


def test_evaluation_mode_experiments_can_precede_any_dataset():
    project = MLOpsWriter(project_name="finetune")
    root = project.add_action("kickoff", "scope the fine-tune")
    baseline = project.add_experiment(
        "granite-docling", parents=root["id"], mode="evaluation", metrics={"cer": 0.19}
    )
    assert baseline["parents"] == [root["id"]]
    assert baseline["data"]["mode"] == "evaluation"

    with pytest.raises(ValueError, match="experiment mode"):
        project.add_experiment("x", parents=root["id"], mode="guessing")


def test_runs_autosave_once_the_project_has_been_saved(tmp_path, writer):
    from mlops import load_project

    writer.save(tmp_path)
    with writer.run("lightgbm", "train") as run:
        run.log_metrics(pr_auc=0.71)

    assert load_project("lineage-demo", tmp_path).experiments[0]["data"]["metrics"] == {"pr_auc": 0.71}


def test_runs_do_not_autosave_before_the_first_save(tmp_path, writer):
    with writer.run("lightgbm", "train") as run:
        run.log_metrics(pr_auc=0.71)
    assert not list(tmp_path.glob("*.json"))


# --------------------------------------------------------------------- lineage

def test_artifacts_link_to_the_run_that_produced_them(writer):
    with writer.run("lightgbm", "train") as run:
        run.log_metrics(pr_auc=0.71)
        artifact = run.log_artifact("model.txt", "s3://bucket/models/lgbm.txt")

    assert artifact["parents"] == ["exp-001"]
    assert artifact["data"]["kind"] == "model"
    assert writer.artifact_nodes == [artifact]

    updated = writer.add_artifact("model.txt", "s3://bucket/models/lgbm-v2.txt", note="retrained")
    assert len(writer.artifact_nodes) == 1
    assert updated["data"]["artifact_ref"].endswith("lgbm-v2.txt")
    assert updated["updated_at"] >= updated["created_at"]


def test_artifacts_validate_kind_reference_and_run(writer):
    with writer.run("lightgbm", "train"):
        pass
    with pytest.raises(ValueError, match="artifact kind"):
        writer.add_artifact("m", "s3://bucket/m.pt", kind="banana")
    with pytest.raises(ValueError):
        writer.add_artifact("m", "ftp://example.com/m.pt")
    with pytest.raises(KeyError, match="exp-404"):
        writer.add_artifact("m", "s3://bucket/m.pt", run_id="exp-404")
    with pytest.raises(ValueError, match="derived from itself"):
        writer.add_artifact("m", "s3://bucket/m.pt", derived_from="m")


def test_lineage_maps_raw_data_through_runs_to_artifacts(writer):
    with writer.run("lightgbm", "train") as run:
        run.log_artifact("model.txt", "s3://bucket/models/lgbm.txt")
    writer.add_artifact("model.onnx", "s3://bucket/models/lgbm.onnx", kind="export", derived_from="model.txt")
    writer.add_artifact("scorer", "https://scoring.internal/fraud", kind="deployment", derived_from="model.onnx")

    graph = writer.graph()
    assert [node["id"] for node in graph["nodes"]] == [
        "act-001", "ds-001", "ds-002", "exp-001", "art-001", "art-002", "art-003",
    ]
    assert graph["edges"] == [
        {"from": "act-001", "to": "ds-001"},
        {"from": "ds-001", "to": "ds-002"},
        {"from": "ds-002", "to": "exp-001"},
        {"from": "exp-001", "to": "art-001"},
        {"from": "art-001", "to": "art-002"},
        {"from": "art-002", "to": "art-003"},
    ]
    assert graph["depths"]["art-003"] == 6


def test_dataset_cannot_derive_from_itself(writer):
    with pytest.raises(ValueError, match="derived from itself"):
        writer.set_dataset("train", "data/loop.csv", derived_from="train")


def test_markdown_export_covers_new_sections(tmp_path, writer):
    writer.edit_action("act-001", details="scoped the project properly")
    with writer.run("lightgbm", "train", parameters={"learning_rate": 0.05}) as run:
        run.log_metrics(pr_auc=0.71)
        run.log_artifact("model.txt", "s3://bucket/models/lgbm.txt")
        run.attach_link("https://example.org/model-card", "model card")

    path = tmp_path / "notes.md"
    writer.to_markdown(path)
    text = path.read_text()
    assert "## Tree" in text and "## Artifacts" in text and "## Attachments" in text
    assert "parameters: {'learning_rate': 0.05}" in text
    assert "ds-002 --> exp-001" in text
    assert "(edited " in text
    assert "link: model card" in text
