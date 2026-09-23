"""No-intercept Ridge policy, with training-only model/threshold selection."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


def design(X, Y):
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.shape != (len(X),) or not len(X) or not X.shape[1]:
        raise ValueError("Expected nonempty X[n,d] and Y[n].")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Non-finite policy inputs.")
    return X, Y


def ridge_weights(X, Y, alpha=5.0):
    X, Y = design(X, Y)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive.")
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def best_threshold(scores, labels):
    """Exact training accuracy optimum; lowest threshold wins ties.

    Includes the all-negative and all-positive classifiers. The original
    exploratory notebook instead searched 300 evenly spaced thresholds.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    if scores.ndim != 1 or len(scores) == 0 or labels.shape != scores.shape:
        raise ValueError("Expected matching nonempty score and label vectors.")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels binary.")
    values, inverse = np.unique(scores, return_inverse=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(values))
    negatives = np.bincount(inverse, weights=1 - labels, minlength=len(values))
    correct = np.r_[labels.sum(), labels.sum() + np.cumsum(negatives - positives)]
    candidates = np.r_[values, np.nextafter(values[-1], np.inf)]
    return float(candidates[np.argmax(correct)])


@dataclass
class RidgePolicy:
    feature_ids: list[int]
    weights: list[float]
    alpha: float
    threshold: float | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if len(self.feature_ids) != len(self.weights) or len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("Policy requires unique feature IDs matching weights.")
        if not self.weights or not np.isfinite(self.weights).all() or self.alpha <= 0:
            raise ValueError("Invalid policy weights/alpha.")

    def predict(self, X, feature_ids):
        if list(feature_ids) != self.feature_ids:
            raise ValueError("Feature order differs from the saved model.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.weights) or not np.isfinite(X).all():
            raise ValueError("Invalid prediction matrix.")
        return X @ np.asarray(self.weights)

    def classify(self, X, feature_ids):
        if self.threshold is None:
            raise ValueError("This attribution policy has no classification threshold.")
        return (self.predict(X, feature_ids) >= self.threshold).astype(int)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"schema_version": 1, **asdict(self)}, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        if obj.pop("schema_version") != 1:
            raise ValueError("Unsupported policy schema.")
        return cls(**obj)


def fit_policy(X, Y, feature_ids, *, alpha=5.0, labels=None, metadata=None):
    X, Y = design(X, Y)
    if len(feature_ids) != X.shape[1]:
        raise ValueError("Feature IDs do not match the design matrix.")
    W = ridge_weights(X, Y, alpha)
    threshold = None if labels is None else best_threshold(X @ W, labels)
    return RidgePolicy(list(map(int, feature_ids)), W.tolist(), float(alpha), threshold, metadata)


def select_alpha(X, Y, candidates, *, seed=42, folds=10):
    """Call ONLY with the training partition. No feature scaling or intercept."""
    from sklearn.model_selection import KFold
    X, Y = design(X, Y)
    if len(X) < folds:
        raise ValueError("Not enough training rows for the requested folds.")
    splits = list(KFold(folds, shuffle=True, random_state=seed).split(X))
    scores = []
    for alpha in candidates:
        mse = [np.mean((X[va] @ ridge_weights(X[tr], Y[tr], alpha) - Y[va]) ** 2) for tr, va in splits]
        scores.append({"alpha": float(alpha), "mean_mse": float(np.mean(mse))})
    if not scores:
        raise ValueError("No alpha candidates.")
    return min(scores, key=lambda row: row["mean_mse"])["alpha"], scores
