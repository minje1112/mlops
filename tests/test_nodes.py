import pytest

from mlops import MLOpsWriter
from mlops.nodes import ancestors, depths, descendants, extract_paths, next_node_id
from mlops.store import safe_filename, store_attachment


@pytest.fixture
def project():
    writer = MLOpsWriter(project_name="finetune-granite")
    writer.add_action("kickoff", "scope the OCR fine-tune", category="planning")
    return writer


# ----------------------------------------------------------------- graph rules

def test_the_first_node_is_the_root_and_takes_no_parents():
    writer = MLOpsWriter(project_name="fresh")
    root = writer.add_node("action", "kickoff")
    assert root["parents"] == []
    assert writer.root is root

    with pytest.raises(ValueError, match="first node in a project is the root"):
        MLOpsWriter(project_name="other").add_node("action", "kickoff", ["act-001"])


def test_every_later_node_needs_a_parent(project):
    with pytest.raises(ValueError, match="at least one parent"):
        project.add_node("note", "floating thought")
    with pytest.raises(KeyError, match="parent node 'nope'"):
        project.add_node("note", "bad parent", ["nope"])
    with pytest.raises(ValueError, match="title must not be empty"):
        project.add_node("note", "   ", ["act-001"])
    with pytest.raises(ValueError, match="node type"):
        project.add_node("banana", "x", ["act-001"])


def test_a_node_can_have_several_parents_and_children(project):
    baseline = project.add_experiment("granite-docling", parents="act-001", mode="evaluation")
    dataset = project.add_dataset("id-instruct", "s3://acme/id/v1.jsonl", parents="act-001")
    finetune = project.add_experiment(
        "granite-lora", parents=[baseline["id"], dataset["id"]], mode="finetune"
    )

    assert finetune["parents"] == [baseline["id"], dataset["id"]]
    assert {node["id"] for node in project.children("act-001")} == {baseline["id"], dataset["id"]}
    assert {node["id"] for node in project.ancestors(finetune)} == {
        "act-001", baseline["id"], dataset["id"]
    }


def test_reparenting_is_checked_for_cycles(project):
    first = project.add_note("first", parents="act-001")
    second = project.add_note("second", parents=first["id"])

    project.edit_node(second, parents=["act-001"])
    assert second["parents"] == ["act-001"]

    with pytest.raises(ValueError, match="descends from"):
        project.edit_node("act-001", parents=[first["id"]])
    with pytest.raises(ValueError, match="its own parent"):
        project.edit_node(first, parents=[first["id"]])


def test_the_root_can_be_edited_and_stays_parentless(project):
    child = project.add_note("first", parents="act-001")

    # The dashboard's edit form posts the parent list it collected, which is
    # empty for the root — that must not be read as "orphan this node".
    edited = project.edit_node("act-001", summary="scope the OCR fine-tune properly", parents=[])
    assert edited["parents"] == []
    assert edited["summary"] == "scope the OCR fine-tune properly"
    assert project.root["id"] == "act-001"

    edited = project.edit_action("act-001", details="revised again")
    assert edited["parents"] == []

    # Clearing the parents of a node that has some is still refused.
    with pytest.raises(ValueError, match="at least one parent"):
        project.edit_node(child, parents=[])


def test_ids_are_readable_and_typed(project):
    project.add_note("a", parents="act-001")
    project.add_dataset("d", "data/d.csv", parents="act-001")
    project.add_experiment("m", parents="act-001")
    assert [node["id"] for node in project.nodes] == ["act-001", "note-001", "ds-001", "exp-001"]
    assert next_node_id("action", project.nodes) == "act-002"


def test_graph_helpers_walk_both_directions():
    nodes = [
        {"id": "a", "parents": []},
        {"id": "b", "parents": ["a"]},
        {"id": "c", "parents": ["a"]},
        {"id": "d", "parents": ["b", "c"]},
    ]
    assert ancestors(nodes, "d") == {"a", "b", "c"}
    assert descendants(nodes, "a") == {"b", "c", "d"}
    assert depths(nodes) == {"a": 0, "b": 1, "c": 1, "d": 2}


def test_editing_stamps_the_last_edited_time(project):
    node = project.add_note("observation", "cyrillic is weak", parents="act-001")
    assert node["updated_at"] == node["created_at"]

    edited = project.edit_node(node, summary="cyrillic and mongolian are weak")
    assert edited["updated_at"] > edited["created_at"]

    with pytest.raises(ValueError, match="at least one field"):
        project.edit_node(node)


# ------------------------------------------------------- dataset versioning

def test_dataset_versions_chain_and_carry_a_split(project):
    first = project.add_dataset("id-instruct", "data/v1.jsonl", split="train", parents="act-001")
    second = project.add_dataset("id-instruct", "data/v2.jsonl", split="train")
    holdout = project.add_dataset("id-holdout", "data/test.jsonl", split="test", parents="act-001")

    assert (first["data"]["version"], second["data"]["version"]) == (1, 2)
    assert second["parents"] == [first["id"]]  # the previous version is always a parent
    assert holdout["data"]["split"] == "test"
    assert project.latest_dataset("id-instruct")["id"] == second["id"]
    assert [node["id"] for node in project.dataset_history("id-instruct")] == [first["id"], second["id"]]

    with pytest.raises(ValueError, match="dataset split"):
        project.add_dataset("bad", "data/x.csv", split="holdout", parents="act-001")


def test_experiments_name_the_datasets_they_used(project):
    train = project.add_dataset("id-instruct", "data/v1.jsonl", split="train", parents="act-001")
    test = project.add_dataset("id-holdout", "data/test.jsonl", split="test", parents="act-001")
    run = project.add_experiment(
        "granite-lora", mode="finetune",
        datasets={"id-instruct": "train", "id-holdout": "test"},
    )
    assert run["data"]["dataset_roles"] == {train["id"]: "train", test["id"]: "test"}
    assert set(run["parents"]) == {train["id"], test["id"]}

    with pytest.raises(KeyError, match="dataset 'missing'"):
        project.add_experiment("x", datasets={"missing": "train"})


# ------------------------------------------------------------- attachments

def test_nodes_carry_links_paths_and_files(project, tmp_path):
    picture = tmp_path / "confusion matrix.png"
    picture.write_bytes(b"\x89PNG fake")
    report = tmp_path / "eval.pdf"
    report.write_bytes(b"%PDF-1.7 fake")

    project.save(tmp_path / "docs")
    node = project.root

    link = project.attach_link(node, "https://example.org/model-card", "model card")
    path = project.attach_path(node, tmp_path, "working folder")
    image = project.attach_file(node, picture)
    pdf = project.attach_file(node, report)

    assert [item["id"] for item in node["attachments"]] == ["att-001", "att-002", "att-003", "att-004"]
    assert (link["type"], link["href"]) == ("link", "https://example.org/model-card")
    assert (path["type"], path["href"]) == ("path", str(tmp_path))
    assert image["type"] == "image" and image["filename"] == "confusion-matrix.png"
    assert pdf["type"] == "pdf" and pdf["size"] == len(b"%PDF-1.7 fake")

    stored = tmp_path / "docs" / "attachments" / "finetune-granite" / "act-001"
    assert sorted(item.name for item in stored.iterdir()) == ["confusion-matrix.png", "eval.pdf"]

    project.remove_attachment(node, image["id"])
    assert [item["id"] for item in node["attachments"]] == ["att-001", "att-002", "att-004"]
    assert not (stored / "confusion-matrix.png").exists()

    with pytest.raises(KeyError, match="att-404"):
        project.remove_attachment(node, "att-404")
    with pytest.raises(ValueError, match="URL with a scheme"):
        project.attach_link(node, "not-a-url")
    with pytest.raises(FileNotFoundError):
        project.attach_file(node, tmp_path / "missing.png")


def test_attachment_filenames_are_sanitised_and_capped(tmp_path):
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my report (final).pdf") == "my-report-final-.pdf"
    with pytest.raises(ValueError):
        safe_filename("../")

    store_attachment("proj", "act-001", "a.txt", b"one", tmp_path)
    second = store_attachment("proj", "act-001", "a.txt", b"two", tmp_path)
    assert second.name == "a-2.txt"  # never silently overwrites

    with pytest.raises(ValueError, match="larger than"):
        store_attachment("proj", "act-001", "big.bin", b"x" * (33 * 1024 * 1024), tmp_path)


# ------------------------------------------------------------- path detection

def test_paths_are_found_in_free_text():
    assert extract_paths("exported to /home/rookie/Documents/Minje/id-craft-ocr/exported.") == [
        "/home/rookie/Documents/Minje/id-craft-ocr/exported"
    ]
    assert extract_paths("see ~/mlops_docs and ./data/raw") == ["~/mlops_docs", "./data/raw"]
    assert extract_paths(r"on windows C:\Users\rookie\data") == [r"C:\Users\rookie\data"]
    assert extract_paths("no paths here", "") == []
    assert extract_paths("https://example.org/a/b") == []
    assert extract_paths("/tmp/a /tmp/a") == ["/tmp/a"]
