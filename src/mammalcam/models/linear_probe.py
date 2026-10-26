"""Logistic regression on frozen embeddings, regularisation picked on validation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)


def fit(x_train, y_train, x_val, y_val, c_grid=C_GRID, seed: int = 0):
    """Return (best model, {C: validation macro-F1}).

    Class weights are balanced so rare species are not ignored; the choice
    of C maximises macro-F1 on the validation split.
    """
    scores, best, best_f1 = {}, None, -1.0
    for c in c_grid:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed),
        ).fit(x_train, y_train)
        f1 = f1_score(y_val, model.predict(x_val), average="macro", labels=np.unique(y_train))
        scores[c] = f1
        if f1 > best_f1:
            best, best_f1 = model, f1
    return best, scores
