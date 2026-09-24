# mlops

A simple, lightweight personal MLOps document writer. A project is a **tree of
nodes**: you create one root action for what you did first, and every later piece
of work names the node(s) it grew out of. That way every experiment carries its
own origin — you can always see what led to it.

- **actions** and **notes** — what you did and what you saw, editable, with last-edited times
- **dataset** nodes — versioned, each with a split (raw/train/validation/test/…), references only
- **experiment** nodes — evaluation, training or fine-tune runs with parameters, metrics, timing and environment
- **artifact** nodes — models, exports, reports, deployments
- **attachments** on any node — pictures, PDFs, links, and folders you can open from the dashboard
- a web dashboard that draws the tree and lets you build it by clicking

Everything is standard library only — no dependencies to install.

## Why

This project is for documenting your own ML workflow end-to-end without storing heavy assets.

## The graph rules

- The **first node you create is the root** and has no parents.
- **Every other node needs at least one parent.** Nothing floats.
- A node can have **any number of parents and children**, so a fine-tune that used
  a baseline run *and* two datasets records all three.

## Quick usage

A typical fine-tuning story — run the stock model first, note what you saw, build
a dataset from it, then fine-tune using both:

```python
from mlops import MLOpsWriter

writer = MLOpsWriter("granite-docling", objective="cut CER below 0.08")

kickoff = writer.add_action("kickoff", "scope the OCR fine-tune", category="planning")

baseline = writer.add_experiment("granite-docling-258M", parents=kickoff["id"],
                                 mode="evaluation", metrics={"cer": 0.19})
writer.attach_file(baseline, "reports/cer-by-field.png", "CER by field")
writer.attach_link(baseline, "https://huggingface.co/ibm-granite/granite-docling-258M")

note = writer.add_note("Cyrillic surnames are the weak spot",
                       "Samples in /home/rookie/Documents/Minje/id-craft-ocr/exported",
                       parents=baseline["id"])

train = writer.add_dataset("id-instruct", "/home/rookie/.../exported",
                           split="train", parents=note["id"])
holdout = writer.add_dataset("id-holdout", "s3://acme/id/test.jsonl",
                             split="test", parents=kickoff["id"])

with writer.run("granite-docling-lora", parents=[baseline["id"]],
                datasets={"id-instruct": "train", "id-holdout": "test"},
                mode="finetune", architecture="LoRA r=16") as run:
    run.log_params(learning_rate=1e-4, rank=16, epochs=3)
    run.log_metrics(cer=0.071)
    run.log_artifact("granite-id-lora", "s3://acme/id/adapters/lora-v1")

writer.save()  # -> mlops_docs/granite-docling.json, picked up by the web dashboard
```

Omitting `parents=` attaches to the root, which keeps short scripts working; pass
it explicitly to record the real origin. Passing `parents=[]` is refused.

## Dataset version control

Each call to `add_dataset` creates the next **version** of that name, and the
previous version automatically becomes its parent — so a dataset's history is
part of the tree:

```python
writer.add_dataset("id-instruct", "data/v1.jsonl", split="train", parents=note["id"])
writer.add_dataset("id-instruct", "data/v2.jsonl")   # v2, parented to v1

writer.latest_dataset("id-instruct")     # the newest version node
writer.dataset_history("id-instruct")    # every version, oldest first
```

Experiments name the datasets they used and the role each played, so it is always
clear which data fed which run:

```python
writer.add_experiment("granite-lora", mode="finetune",
                      datasets={"id-instruct": "train", "id-holdout": "test"})
```

## Attachments

Any node takes pictures, PDFs, links and filesystem paths:

```python
writer.attach_file(node, "plots/confusion.png")   # copied into <docs>/attachments/
writer.attach_link(node, "https://example.org/model-card", "model card")
writer.attach_path(node, "~/Documents/Minje/id-craft-ocr")  # left where it is
writer.remove_attachment(node, "att-001")
```

Uploaded files are copied into `<docs root>/attachments/<project>/<node>/`, so
they survive the original being moved. Images and PDFs preview in the dashboard.

## Folder links

Any filesystem path written into a node's notes — plus every `path` attachment —
becomes a clickable chip in the dashboard. Clicking it opens that folder in your
desktop file manager (`xdg-open`, or the file revealed in its folder where the
file manager supports it).

Because that launches a desktop program, it is only offered when the server is
bound to localhost, and `mlops-web --no-open` turns it off entirely.

## Experiment tracking and metadata logging

`writer.run(...)` is a context manager around one training run. It assigns a run
id (`exp-001`), times the run, and captures the Python version, platform and git
commit automatically, so every experiment is reproducible without extra work:

```python
with writer.run("lightgbm", "train", architecture="gbdt, 64 leaves") as run:
    run.log_params(learning_rate=0.05, num_leaves=64)   # hyperparameters
    run.log_params({"optimizer": "adam"})               # dict form works too
    run.set_architecture("gbdt, 128 leaves")            # revise mid-run
    run.log_metrics(pr_auc=0.71, ndcg_at_10=0.42)       # evaluation metrics
    run.log_evaluation(y_true, y_pred)                  # computed for you
    run.log_artifact("model.txt", "s3://bucket/models/lgbm.txt")
```

- **Parameters** — `log_params` takes keywords or a dict; hyperparameters,
  learning rates and any other setting land in the run's `parameters`.
- **Metrics** — `log_metrics` records what you measured. `log_evaluation(y_true,
  y_pred)` derives accuracy, precision, recall and f1 for you (binary by default,
  macro-averaged for multiclass; use `prefix="val_"` to namespace them).
- **Status** — a run that raises is stored as `failed` with the exception message
  and the exception is re-raised, so failed experiments are documented too.
- **Autosave** — once a project has been saved once, finished runs persist
  automatically. Pass `autosave=False` to opt out, or `autosave=True` to force it.

`writer.add_training_run(...)` still works for recording a run you already have
metrics for. Note that both paths store settings under the run's `parameters`
key (the `hyperparameters=` argument is still accepted and merged into it).

### Adding to a run after it finished

Scored a checkpoint the next day? Re-open the experiment and log into it — the
same handle, with merge semantics, so new numbers add to what is there:

```python
run = writer.log_into("exp-001")
run.log_metrics(holdout_tag_acc=0.77)
run.log_artifact("doctag-v6", "/path/to/output_dir")
writer.save()
```

Use `writer.edit_node("exp-001", metrics={...})` instead when you mean to
*replace* a field outright — that is what the dashboard's Edit button does.

## Importing a Weights & Biases run

A run that crashed records only the exception, while W&B still holds every metric
logged before it died. Pull one into a node:

```bash
# from the local run directory — no network, no wandb package needed
python3 -m mlops.integrations.wandb wandb/run-20260902_111641-niqajvlo \
    --project granite-docling --parents ds-001

# or through the API (needs the wandb package). entity/project/run_id, the run
# URL, and the path copied out of the address bar all work:
python3 -m mlops.integrations.wandb my-team/Docling-Restart/19hgozfy \
    --project granite-docling --parents ds-003,exp-003
```

The entity is whoever **owns** the run: for a team run that is the team, not your
username, so copy it from the run's URL rather than assuming it is your login.

A run only has a local directory on the machine that produced it. Point at a
`wandb/run-*` path that does not exist and the importer says so, instead of
reading your filesystem path as a W&B run path.

```python
from mlops import load_project
from mlops.integrations.wandb import import_run

writer = load_project("granite-docling")
import_run(writer, "wandb/run-20260902_111641-niqajvlo", parents="ds-001")
writer.save()
```

What comes across: the run's command line as parameters, its summary as metrics,
runtime as duration, and Python / OS / GPU / CUDA / git commit as environment. The
W&B run URL is attached as a link when the source knows it.

Re-importing the same run **updates its node** rather than adding a second one, so
a run can be pulled again once it finishes. A local directory has no authoritative
final state, so the importer only claims `failed` when the console log ends in an
exception, and otherwise leaves the status alone — a status your training script
recorded first-hand is never overwritten by a guess. Pass `--mode` to override the
guess from `--lora`, and `--include-config` to also record the run's full config
rather than just its command line.

Throughput counters the Trainer logs beside real results — `*_runtime`,
`*_samples_per_second`, `*_steps_per_second`, `total_flos` — are dropped, since a
"best total_flos" across runs means nothing. `--keep-throughput` retains them.

## The side note

Every project has one free-text scratchpad that is **not** a node: the thinking
that has no place in the graph yet, a pasted metric block, a reminder for next
time. It lives in the project JSON as `draft`, so it travels with the document
and reaches the markdown export, but it never appears in the tree, the counts or
the lineage.

In the dashboard it is the sticky note beside the root node — tap it to read or
write, Ctrl/Cmd+Enter to save. A dot on the sticky means there is something in it.

```python
writer.set_draft("mixed30 beat textonly.\n\nnext: try r=64")
writer.draft, writer.draft_updated_at
writer.set_draft("")          # clearing it also drops the timestamp
```

## Walking the graph

```python
writer.root                       # the origin node
writer.graph()                    # {"nodes": [...], "edges": [...], "depths": {...}}
writer.ancestors("exp-002")       # everything this experiment grew out of
writer.children("act-001")        # what came next
writer.experiments                # or .dataset_versions / .artifact_nodes / .action_nodes
```

Artifacts chain the same way, so a model can lead to an export and then to a
deployment:

```python
writer.add_artifact("model.onnx", "s3://bucket/models/m.onnx", kind="export",
                    derived_from="model.txt")
writer.add_artifact("scorer", "https://scoring.internal/fraud", kind="deployment",
                    derived_from="model.onnx")
```

Artifact kinds are `model`, `export`, `report` and `deployment`. Registering an
artifact under an existing name updates it in place.

## Editing nodes

Every node carries a creation time and a last-edited time:

```python
writer.edit_node("note-001", summary="Cyrillic and Mongolian surnames are weak")
writer.edit_node("exp-002", parents=["exp-001", "ds-002"])   # re-home it
writer.get_node("note-001")["updated_at"]   # later than ["created_at"]

# actions keep their own shorthand
writer.add_action("baseline review", "compared against the production ruleset")
writer.edit_action("act-002", details="compared against the ruleset; +9pts recall")
```

Re-parenting is checked against the descendants of the node, so the graph can
never form a cycle.

## Web dashboard

`writer.save()` stores each project as one JSON file under `mlops_docs/`. The
dashboard reads that folder and shows every project and its trackings in a browser:

```bash
python3 -m mlops.web              # or: mlops-web
python3 -m mlops.web --root mlops_docs --port 8787
```

Then open http://127.0.0.1:8787. The left column lists your projects (filterable,
most recently active first) with a **+ New** button to create one.

Selecting a project draws its **tree**, top-down from the root. The layout is
layered: an edge that spans more than one row is routed down its own reserved
lane instead of cutting across the boxes in between, rows are ordered to reduce
crossings, and a node with several parents receives them at separate points along
its edge rather than at a single pinch.

Edges are drawn as right angles rather than sweeping curves, which is far easier
to follow where several share the space between two rows. An edge that crosses
more than one row is real but secondary, so it is drawn thinner, paler and
behind — hover any edge to light its whole path, or select a node to light its
ancestry. Nothing is hidden. Click any node to
select it: its whole origin chain lights up, everything unrelated dims, and a
panel opens with its fields, notes, attachments, and the nodes it came from and
led to. From there:

- **+ Add node underneath** creates a child with that node pre-filled as parent
  (Ctrl/Cmd-click to pick several parents — a fine-tune often has three)
- **Edit** revises the node and stamps its last-edited time
- **Attach** adds a picture, PDF, link or folder
- 📁 chips — in notes and on path attachments — open that folder in your file manager

Below the tree are flat views of the same nodes: metric summary, the experiments
table, dataset versions with their splits, and artifacts. Writes go through the
same document files, so a project edited in the browser is the same one
`load_project()` returns.

Saving again updates the same file, so refreshing the page shows new runs. To keep
tracking a project from a later session:

```python
from mlops import load_project

writer = load_project("granite-docling")
with writer.run("granite-docling-lora", parents=["exp-002"],
                datasets={"id-instruct": "train"}, mode="finetune") as run:
    run.log_metrics(cer=0.065)
writer.save()
```

Projects written before the tree existed are converted automatically when they
load: the earliest action becomes the root, and the lineage that was implied by
`derived_from` and `run_id` becomes real parent links. The file on disk is
rewritten in node form the next time you save.

The page is re-read from disk on every request, but the routes live in the
running process — so a dashboard newer than the server would otherwise fail with
a bare "not found". The server reports an API version and the page shows a banner
telling you to restart when it is behind.

The server is standard-library only and binds to localhost by default. `--port 0`
picks a free port; `--host 0.0.0.0` exposes it on your network — note that it can
write to your documents, so only do that on a network you trust.

## Using the tracker from another repo

**1. Install it once**, so `import mlops` works outside this folder. Ubuntu's system
Python is externally managed (PEP 668), so plain `pip install` is refused. Install
into your user site instead — `mlops` has no dependencies, so nothing system-wide
is touched beyond a link into `~/.local`:

```bash
pip install --user --break-system-packages -e ~/Documents/Minje/mlops
```

`-e` means edits made here apply everywhere immediately. If a repo has its own
virtualenv, install into that venv explicitly rather than relying on an activated
prompt:

```bash
cd ~/path/to/your-ml-repo
python3 -m venv .venv                                # only if it has no venv yet
.venv/bin/pip install -e ~/Documents/Minje/mlops
```

**2. Point every repo at one docs folder** by setting `MLOPS_DOCS`, so a single
dashboard shows all your projects instead of one per repo:

```bash
echo 'export MLOPS_DOCS="$HOME/mlops_docs"' >> ~/.bashrc && source ~/.bashrc
```

**3. Track from anywhere.** In any repo, any script or notebook:

```python
from mlops import MLOpsWriter, load_project

writer = MLOpsWriter("recsys-reranker", objective="raise NDCG@10 to 0.42")
writer.set_dataset("clicks", "s3://acme-ml/recsys/2026-08/clicks.parquet")
writer.add_training_run("xgb-ranker", "clicks", metrics={"ndcg@10": 0.39})
writer.save()          # -> $MLOPS_DOCS/recsys-reranker.json

writer = load_project("recsys-reranker")   # resume later, from any directory
writer.add_training_run("lambdamart", "clicks", metrics={"ndcg@10": 0.43})
writer.save()
```

**4. View everything** with one dashboard, launched from any directory:

```bash
mlops-web
```

Without `MLOPS_DOCS`, `save()` writes to `./mlops_docs` relative to the current
directory, giving each repo its own set of documents; point the dashboard at one
with `mlops-web --root ~/path/to/your-ml-repo/mlops_docs`. An explicit `root=`
argument always wins over the environment variable.
