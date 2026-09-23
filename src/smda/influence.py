"""First-order derivative in paper Section 3.3 / Appendix A.3."""
from dataclasses import dataclass
import numpy as np
from .policy import design, ridge_weights


@dataclass
class Influence:
    delta_w_x: np.ndarray
    delta_w_y: np.ndarray
    delta_w: np.ndarray
    score: float
    weight_alignment_by_feature: np.ndarray


def linearized_features(delta_h, h_base, W_enc, b_enc):
    """Paper's dense ReLU proxy; W_enc has shape [hidden, selected_features].

    This deliberately follows the paper equation, not the complete BatchTopK
    checkpoint encoder (which also has input centering and sparse gating).
    """
    dh, h, w, b = [np.asarray(x, dtype=np.float64) for x in (delta_h, h_base, W_enc, b_enc)]
    if h.ndim != 2 or dh.shape != h.shape or w.ndim != 2 or w.shape[0] != h.shape[1] or b.shape != (w.shape[1],):
        raise ValueError("SAE input dimensions do not match.")
    if not all(np.isfinite(x).all() for x in (dh, h, w, b)):
        raise ValueError("Non-finite SAE input.")
    return (h @ w + b > 0) * (dh @ w)


def decompose_influence(X, Y, delta_X, delta_Y, *, alpha=5.0):
    X, Y = design(X, Y)
    dx, dy = np.asarray(delta_X, dtype=np.float64), np.asarray(delta_Y, dtype=np.float64)
    if dx.shape != X.shape or dy.shape != Y.shape or not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError("Invalid perturbation dimensions/values.")
    W = ridge_weights(X, Y, alpha)
    U = X.T @ X + alpha * np.eye(X.shape[1])
    delta_U = dx.T @ X + X.T @ dx
    wx = np.linalg.solve(U, dx.T @ Y - delta_U @ W)
    wy = np.linalg.solve(U, X.T @ dy)
    dw = wx + wy
    contributions = dw * W
    return Influence(wx, wy, dw, float(contributions.sum()), contributions)
