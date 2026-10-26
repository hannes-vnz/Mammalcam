"""Classification metrics shared by the baseline and the crop classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def summary(y_true, y_pred, classes: list[str]) -> dict:
    """Accuracy and macro-F1 over the given classes.

    Predictions outside ``classes`` (e.g. "no detection") count as wrong.
    """
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
    }


def per_class(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return pd.DataFrame({"support": n, "precision": p, "recall": r, "f1": f}, index=pd.Index(classes, name="class"))


def confusion(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    """Rows: true class, columns: predicted class (plus any extra predicted labels)."""
    extra = sorted(set(np.asarray(y_pred)) - set(classes))
    labels = list(classes) + extra
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm[: len(classes)], index=pd.Index(classes, name="true"), columns=labels)
