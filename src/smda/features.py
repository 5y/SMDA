"""Paper's contrastive feature selection and examples, without a paid labeling API."""
import numpy as np


def select_features(activations, harmful, *, threshold=0.1):
    X, labels = np.asarray(activations, dtype=float), np.asarray(harmful)
    if X.ndim != 2 or labels.shape != (len(X),) or not np.isfinite(X).all() or np.any(X < 0):
        raise ValueError("Expected finite nonnegative SAE activations and matching labels.")
    if threshold < 0 or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Need both classes and a nonnegative threshold.")
    h = labels.astype(bool)
    diff = X[h].mean(0) - X[~h].mean(0)
    # Strict > matches 'exceeds 0.1' in the paper. Counts are measured, not forced.
    ids = np.flatnonzero(np.abs(diff) > threshold)
    return [{"feature_idx": int(i), "mean_diff": float(diff[i]),
             "direction": "harmful" if diff[i] > 0 else "harmless"} for i in ids]


def labeling_examples(activations, feature_id, same_direction_ids, *, contrastive_n=6, extremes_n=8):
    X = np.asarray(activations, dtype=float)
    peers = [i for i in same_direction_ids if i != feature_id]
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("Invalid activation matrix.")
    if not peers or not all(0 <= i < X.shape[1] for i in [feature_id, *peers]):
        raise ValueError("Need a valid target feature and at least one same-direction peer.")
    if contrastive_n < 1 or extremes_n < 1:
        raise ValueError("Example counts must be positive.")
    contrast = X[:, feature_id] - X[:, peers].mean(1)
    return {"contrastive": np.argsort(-contrast, kind="stable")[:contrastive_n].tolist(),
            "top": np.argsort(-X[:, feature_id], kind="stable")[:extremes_n].tolist(),
            "bottom": np.argsort(X[:, feature_id], kind="stable")[:extremes_n].tolist()}
