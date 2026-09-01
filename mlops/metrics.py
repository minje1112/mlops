"""Dependency-free evaluation metrics, so every run can log them automatically."""

from __future__ import annotations

from typing import Any, Sequence


def _pairs(y_true: Sequence[Any], y_pred: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    truth, predicted = list(y_true), list(y_pred)
    if len(truth) != len(predicted):
        raise ValueError(f"y_true has {len(truth)} items but y_pred has {len(predicted)}")
    if not truth:
        raise ValueError("cannot evaluate an empty prediction set")
    return truth, predicted


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _per_label(truth: list[Any], predicted: list[Any], label: Any) -> tuple[float, float]:
    true_positive = sum(1 for t, p in zip(truth, predicted) if t == label and p == label)
    predicted_positive = sum(1 for p in predicted if p == label)
    actual_positive = sum(1 for t in truth if t == label)
    return _ratio(true_positive, predicted_positive), _ratio(true_positive, actual_positive)


def evaluate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any | None = None,
    average: str | None = None,
) -> dict[str, float]:
    """accuracy / precision / recall / f1 for a set of predictions.

    ``average`` defaults to ``"binary"`` for two-class problems and ``"macro"``
    otherwise. For binary problems the positive class defaults to the highest
    label (``1`` for ``{0, 1}``); pass ``positive_label`` to choose another.
    """
    truth, predicted = _pairs(y_true, y_pred)
    labels = sorted(set(truth) | set(predicted), key=str)
    average = average or ("binary" if len(labels) <= 2 else "macro")

    if average == "binary":
        target = positive_label if positive_label is not None else labels[-1]
        if target not in labels:
            raise ValueError(f"positive_label {target!r} does not appear in the predictions")
        precision, recall = _per_label(truth, predicted, target)
    elif average == "macro":
        scores = [_per_label(truth, predicted, label) for label in labels]
        precision = sum(score[0] for score in scores) / len(scores)
        recall = sum(score[1] for score in scores) / len(scores)
    else:
        raise ValueError("average must be 'binary' or 'macro'")

    accuracy = _ratio(sum(1 for t, p in zip(truth, predicted) if t == p), len(truth))
    f1 = _ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "support": float(len(truth)),
    }
