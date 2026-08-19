# mlops

A simple, lightweight personal MLOps document writer for tracking:

- project objective and operations you performed
- dataset references (hyperlinks/paths only, no dataset files)
- dataset change history when a dataset is updated
- model training runs (model, dataset used, metrics, hyperparameters)
- practical ML engineer utilities/checklists

## Why

This project is for documenting your own ML workflow end-to-end without storing heavy assets.

## Quick usage

```python
from mlops import MLOpsWriter

writer = MLOpsWriter("fraud-detection", objective="increase PR-AUC")
writer.add_action("data cleaning", "removed duplicate sessions")
writer.set_dataset("train", "s3://my-bucket/fraud/v1/train.parquet")
writer.log_dataset_change("train", "feature_engineering", "added rolling transaction features")
writer.add_training_run(
    "lightgbm",
    "train",
    metrics={"pr_auc": 0.71},
    hyperparameters={"num_leaves": 64},
)
writer.suggest_utilities()
writer.to_markdown("project_notes.md")
writer.to_json("project_notes.json")
```
