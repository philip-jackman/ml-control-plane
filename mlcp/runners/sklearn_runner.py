"""A small scikit-learn classification runner.

Trains a LogisticRegression on a deterministic synthetic dataset and reports
test accuracy. scikit-learn (and numpy) are optional dependencies, so they are
imported lazily inside `run` — importing this module never requires the extra.
Install it with: pip install ml-control-plane[sklearn]
"""

from __future__ import annotations

from typing import Any

from ..models import RunResult
from ..runner import Runner


class SklearnClassifierRunner(Runner):
    """Fits a logistic-regression classifier on a deterministic synthetic
    dataset and returns test accuracy. Fast, dependency-light, reproducible."""

    name = "sklearn_classifier"

    def run(self, params: dict[str, Any]) -> RunResult:
        try:
            import numpy as np  # noqa: F401  (kept for explicit dependency intent)
            from sklearn.datasets import make_classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn extra not installed; run: pip install ml-control-plane[sklearn]"
            ) from exc

        n_samples = int(params.get("n_samples", 200))
        n_features = int(params.get("n_features", 8))
        n_informative = int(params.get("n_informative", 4))
        test_size = float(params.get("test_size", 0.25))
        seed = int(params.get("seed", 0))

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=seed,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        return RunResult(
            ok=True,
            metrics={
                "accuracy": float(acc),
                "n_train": float(len(X_train)),
                "n_test": float(len(X_test)),
            },
            outputs={"n_features": n_features},
        )
