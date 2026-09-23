"""SMDA research implementation. GPU dependencies are optional."""
from .policy import RidgePolicy, fit_policy
from .influence import decompose_influence, linearized_features

__version__ = "0.1.0"
__all__ = ["RidgePolicy", "fit_policy", "decompose_influence", "linearized_features"]
