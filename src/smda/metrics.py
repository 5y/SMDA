"""Keep soft prefix-probability metrics separate from judged generations."""
import numpy as np


def soft_metrics(probabilities, harmful):
    p, h = np.asarray(probabilities, float), np.asarray(harmful)
    if p.shape != h.shape or p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected aligned probabilities in [0,1].")
    if not np.isin(h, [0, 1]).all() or len(np.unique(h)) != 2:
        raise ValueError("Both harmful and harmless examples are required.")
    h = h.astype(bool)
    tp, fp = float(p[h].sum()), float(p[~h].sum())
    recall = float(p[h].mean())
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {"correctness": float(np.where(h, p, 1-p).mean()), "precision": precision,
            "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}


def classification_metrics(predictions, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    pred, y = np.asarray(predictions), np.asarray(labels)
    if pred.shape != y.shape or pred.ndim != 1 or not len(y) or not np.isin(pred, [0,1]).all() or not np.isin(y, [0,1]).all():
        raise ValueError("Expected aligned nonempty binary vectors.")
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(p), "recall": float(r), "f1": float(f)}
