import json

import pytest

from mlops import MLOpsWriter
from mlops.utils import next_experiment_id, summarize_metrics


def test_tracks_actions_datasets_changes_and_training_runs(tmp_path):
    writer = MLOpsWriter(project_name="churn-prediction", objective="improve recall")

    action = writer.add_action("ingest data", "loaded source table")
    assert action["title"] == "ingest data"

    writer.set_dataset("train", "s3://bucket/churn/v1/train.parquet")
    writer.log_dataset_change("train", "filtering", "Removed null target rows")
    writer.set_dataset("train", "s3://bucket/churn/v2/train.parquet")

    run = writer.add_training_run(
        "xgboost",
        "train",
        metrics={"accuracy": 0.91, "recall": 0.82},
        hyperparameters={"max_depth": 6},
    )
    assert run["dataset_ref"].endswith("/v2/train.parquet")
    assert len(writer.datasets["train"]["changes"]) == 2

    writer.suggest_utilities()
    assert "suggested_utilities" in writer.utilities

    json_path = tmp_path / "project.json"
    md_path = tmp_path / "project.md"
    writer.to_json(json_path)
    writer.to_markdown(md_path)

    payload = json.loads(json_path.read_text())
    assert payload["project_name"] == "churn-prediction"
    assert "## Datasets" in md_path.read_text()


def test_rejects_invalid_dataset_reference():
    writer = MLOpsWriter(project_name="test")
    with pytest.raises(ValueError):
        writer.set_dataset("bad", "not-a-valid-ref")


def test_small_utility_helpers():
    assert next_experiment_id([]) == "exp-001"
    summary = summarize_metrics(
        [
            {"metrics": {"f1": 0.7}},
            {"metrics": {"f1": 0.9}},
        ],
        "f1",
    )
    assert summary == {"count": 2.0, "best": 0.9, "average": 0.8}
