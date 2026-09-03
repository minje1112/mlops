import json

import pytest

from mlops import MLOpsWriter
from mlops.integrations.wandb import (
    _from_api_run,
    _normalise_path,
    argv_to_params,
    from_local,
    import_run,
    import_run_data,
)


def make_run(tmp_path, run_id="niqajvlo", args=None, summary=None, log="epoch 1 done\nSaving model\n"):
    """A minimal copy of the layout wandb writes under wandb/run-*/files."""
    files = tmp_path / f"run-20260902_111641-{run_id}" / "files"
    files.mkdir(parents=True)
    (files / "wandb-metadata.json").write_text(json.dumps({
        "program": "/home/rookie/train_origin.py",
        "args": args if args is not None else ["--lora", "--bf16", "--lr", "2e-5", "--epochs", "10"],
        "python": "CPython 3.12.3",
        "os": "Linux-6.17.0-40-generic-x86_64-with-glibc2.39",
        "host": "rookie-B650M-H-M-2",
        "git": {"remote": "https://github.com/x/y.git", "commit": "1c2d2cbbb0d3ad5a00f38b0535e3851fa994d391"},
        "gpu": "NVIDIA GeForce RTX 3090", "gpu_count": 1, "cudaVersion": "13.0",
    }), encoding="utf-8")
    (files / "wandb-summary.json").write_text(json.dumps(
        summary if summary is not None else {"train/loss": 0.1378, "_step": 81, "_runtime": 1905}
    ), encoding="utf-8")
    (files / "output.log").write_text(log, encoding="utf-8")
    return files.parent


CRASH_LOG = (
    '  File "/x/torch/autograd/graph.py", line 865, in _engine_run_backward\n'
    "    return Variable._execution_engine.run_backward(\n"
    "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx(...)`\n"
)


@pytest.fixture
def project():
    writer = MLOpsWriter(project_name="granite-docling")
    root = writer.add_action("kickoff", "scope the finetune")
    writer.add_dataset("text-only", "/data/text_only", split="train", parents=root["id"])
    return writer


# ------------------------------------------------------------------- parsing

def test_command_line_becomes_typed_parameters():
    assert argv_to_params(["--lora", "--lr", "2e-5", "--epochs", "10", "--bf16"]) == {
        "lora": True, "lr": 2e-05, "epochs": 10, "bf16": True,
    }
    assert argv_to_params(["--out-dir", "run1"]) == {"out_dir": "run1"}
    assert argv_to_params([]) == {}


def test_local_run_is_read_without_network(tmp_path):
    run = from_local(make_run(tmp_path))
    assert run.id == "niqajvlo"
    assert run.duration() == 1905.0
    assert run.metrics() == {"train/loss": 0.1378}      # _step and _runtime dropped
    assert run.parameters() == {"lora": True, "bf16": True, "lr": 2e-05, "epochs": 10}
    assert run.environment() == {
        "python": "CPython 3.12.3",
        "platform": "Linux-6.17.0-40-generic-x86_64-with-glibc2.39",
        "git_commit": "1c2d2cb",
        "gpu": "NVIDIA GeForce RTX 3090",
        "cuda": "13.0",
        "host": "rookie-B650M-H-M-2",
    }


def test_a_crash_is_detected_from_the_console_log(tmp_path):
    crashed = from_local(make_run(tmp_path / "a", log=CRASH_LOG))
    assert crashed.status == "failed"
    assert crashed.error.startswith("RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED")

    # A run that simply ended has no authoritative state on disk, so none is claimed.
    clean = from_local(make_run(tmp_path / "b"))
    assert clean.status is None and clean.error == ""

    # An exception mentioned mid-log is not the run's cause of death.
    noisy = from_local(make_run(tmp_path / "c", log="ValueError: retrying\nrecovered\ndone\n"))
    assert noisy.status is None


def test_run_name_and_project_come_from_the_command_line(tmp_path):
    run = from_local(make_run(tmp_path, args=[
        "--wandb_project", "Docling-Restart", "--wandb_run_name", "all_synthetic_restart_textonly",
    ]))
    assert (run.name, run.project) == ("all_synthetic_restart_textonly", "Docling-Restart")


# ------------------------------------------------------------------ importing

def test_importing_creates_an_experiment_node_under_its_dataset(tmp_path, project):
    node = import_run(project, make_run(tmp_path, log=CRASH_LOG), parents="ds-001")

    data = node["data"]
    assert node["parents"] == ["ds-001"]
    assert node["type"] == "experiment"
    assert data["mode"] == "finetune"                    # --lora implies a finetune
    assert data["status"] == "failed"
    assert data["duration_seconds"] == 1905.0
    assert data["metrics"] == {"train/loss": 0.1378}
    assert data["parameters"]["lr"] == 2e-05
    assert data["error"].startswith("RuntimeError: CUDA")
    assert data["environment"]["gpu"] == "NVIDIA GeForce RTX 3090"
    assert data["wandb"]["id"] == "niqajvlo"


def test_reimporting_updates_the_same_node(tmp_path, project):
    source = make_run(tmp_path)
    first = import_run(project, source, parents="ds-001")
    again = import_run(project, source, parents="ds-001")
    assert first["id"] == again["id"]
    assert len(project.experiments) == 1


def test_import_never_overwrites_a_status_the_source_does_not_know(tmp_path, project):
    node = import_run(project, make_run(tmp_path), parents="ds-001")
    project.edit_node(node, status="failed", error="RuntimeError: recorded by the training script")

    import_run(project, make_run(tmp_path / "second"), parents="ds-001")   # same run id
    assert node["data"]["status"] == "failed"
    assert node["data"]["error"] == "RuntimeError: recorded by the training script"


def test_explicit_mode_and_title_win(tmp_path, project):
    node = import_run(project, make_run(tmp_path), parents="ds-001",
                      mode="evaluation", title="baseline sweep")
    assert (node["data"]["mode"], node["title"]) == ("evaluation", "baseline sweep")


def test_import_obeys_the_parent_rule(tmp_path, project):
    with pytest.raises(ValueError, match="at least one parent"):
        import_run(project, make_run(tmp_path), parents=[])
    with pytest.raises(KeyError, match="parent node 'nope'"):
        import_run(project, make_run(tmp_path / "x"), parents="nope")


# ------------------------------------------------------------------ api shape

class StubRun:
    id, name, entity, project = "abc123", "all_synthetic_restart", "mmsuren0909", "Docling-Restart"
    url = "https://wandb.ai/mmsuren0909/Docling-Restart/runs/abc123"
    state, notes = "crashed", "ran out of memory"
    summary = {"train/loss": 0.2, "_runtime": 60, "_step": 9, "flagged": True}
    config = {"lr": 3e-4, "_wandb": {"x": 1}}
    metadata = {"args": ["--lora"], "python": "CPython 3.12.3"}


def test_throughput_counters_are_dropped_from_metrics():
    class Run(StubRun):
        summary = {
            "eval/loss": 0.033, "train/loss": 0.0329, "train/global_step": 5040,
            "eval/runtime": 447.4, "train_runtime": 36392.4, "total_flos": 1.3e17,
            "eval/samples_per_second": 2.25, "train_steps_per_second": 0.138,
        }
    run = _from_api_run(Run())
    assert set(run.metrics()) == {"eval/loss", "train/loss", "train/global_step"}
    assert len(run.metrics(keep_throughput=True)) == 8


def test_api_run_maps_onto_the_same_shape():
    run = _from_api_run(StubRun())
    assert run.status == "failed"                      # crashed -> failed
    assert run.metrics() == {"train/loss": 0.2}        # booleans and _keys dropped
    assert run.duration() == 60.0
    assert run.parameters() == {"lora": True}
    assert run.parameters(include_config=True) == {"lora": True, "lr": 3e-4}


def test_api_import_attaches_the_run_url(project):
    node = import_run_data(project, _from_api_run(StubRun()), parents="ds-001")
    assert [(a["type"], a["href"]) for a in node["attachments"]] == [("link", StubRun.url)]
    assert node["data"]["wandb"]["url"] == StubRun.url


@pytest.mark.parametrize("given, expected", [
    ("me/proj/abc123", "me/proj/abc123"),
    ("me/proj/runs/abc123", "me/proj/abc123"),
    # the path people copy out of the address bar, with a leading slash
    ("/mmsuren0909-rookie-systems/Docling-Restart/runs/atnafe6d",
     "mmsuren0909-rookie-systems/Docling-Restart/atnafe6d"),
    ("https://wandb.ai/me/proj/runs/abc123", "me/proj/abc123"),
    ("https://wandb.ai/me/proj/runs/abc123?workspace=user", "me/proj/abc123"),
    ("wandb.ai/me/proj/runs/abc123", "me/proj/abc123"),
    ("  me/proj/abc123  ", "me/proj/abc123"),
    ("me/runs/abc123", "me/runs/abc123"),          # a project actually named "runs"
])
def test_run_paths_are_normalised(given, expected):
    assert _normalise_path(given) == expected


@pytest.mark.parametrize("bad", ["justaname", "me/proj", "https://wandb.ai/me", "", "   "])
def test_bad_run_paths_are_rejected(bad):
    with pytest.raises(ValueError):
        _normalise_path(bad)
