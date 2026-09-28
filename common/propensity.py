"""Propensity estimation — code 1.1 / common.

Estimate the **nominal** (observed-data) propensity model ê(X) = P(T = k | X) from the
sample (X, T) and read off the quantities the downstream policy optimisers need.

Convention (project-wide):
  * We estimate P(T = k | X) from the data via (multinomial) logistic regression — the
    *nominal* propensity that ignores the unobserved confounder S. The sensitivity model
    later inflates these by Γ; that inflation is NOT done here.
  * Probabilities are clipped to [clip, 1 - clip] so the inverse weights stay finite
    (a practical positivity / overlap safeguard).
  * Arms are indexed 0 … K-1; arm 0 is the control by convention. K = 2 (binary
    treatment) is just the special case handled by :func:`propensity_binary`.

Public API
----------
    fit_propensity_model(X, T)        -> fitted sklearn LogisticRegression
    propensity_matrix(X, T[, K])      -> (n, K)  P(T=k | X_i), clipped & row-normalised
    factual_propensity(P, T)          -> (n,)    P(T=T_i | X_i)  (the observed-arm column)
    propensity_binary(X, T)           -> (n,)    ê(X_i) = P(T=1 | X_i)  (K = 2 convenience)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

__all__ = [
    "fit_propensity_model",
    "propensity_matrix",
    "factual_propensity",
    "propensity_binary",
]


def fit_propensity_model(X: np.ndarray, T: np.ndarray, **kwargs: Any) -> LogisticRegression:
    """Fit a (multinomial) logistic propensity model ``T ~ X``.

    Defaults: ``solver="lbfgs"``, ``max_iter=2000``. Under lbfgs, modern scikit-learn
    treats a multiclass target as multinomial by default, so we deliberately do **not**
    pass the deprecated ``multi_class`` argument. Override any default via ``**kwargs``
    (e.g. ``C=`` for the inverse-regularisation strength).
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    params: dict[str, Any] = {"solver": "lbfgs", "max_iter": 2000}
    params.update(kwargs)
    return LogisticRegression(**params).fit(X, T)


def propensity_matrix(
    X: np.ndarray,
    T: np.ndarray,
    n_arms: Optional[int] = None,
    *,
    clip: float = 1e-3,
    model: Optional[LogisticRegression] = None,
) -> np.ndarray:
    """Return the ``(n, K)`` propensity matrix ``P(T = k | X_i)``.

    Columns are aligned to arms ``0 … K-1`` (so column ``k`` is always arm ``k`` even if
    the fitted model's ``classes_`` are in another order or omit an absent arm). The
    matrix is clipped to ``[clip, 1-clip]`` and then renormalised so each row sums to 1.

    Parameters
    ----------
    n_arms : inferred as ``T.max() + 1`` when ``None``.
    model  : reuse an already-fitted model instead of refitting (e.g. to score a held-out
             set with the train-fitted propensity).
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    if n_arms is None:
        n_arms = int(T.max()) + 1
    if model is None:
        model = fit_propensity_model(X, T)

    proba = model.predict_proba(X)
    P = np.full((X.shape[0], n_arms), clip, dtype=float)
    for col, cls in enumerate(model.classes_):
        P[:, int(cls)] = proba[:, col]
    P = np.clip(P, clip, 1.0 - clip)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def factual_propensity(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pick the observed-arm probability ``P(T = T_i | X_i)`` from a ``(n, K)`` matrix."""
    P = np.asarray(P, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    if P.shape[0] != T.shape[0]:
        raise ValueError("P and T must have the same number of rows.")
    return P[np.arange(P.shape[0]), T]


def propensity_binary(
    X: np.ndarray, T: np.ndarray, *, clip: float = 1e-3, model: Optional[LogisticRegression] = None
) -> np.ndarray:
    """Convenience for binary treatment: ``ê(X_i) = P(T = 1 | X_i)`` (T ∈ {0, 1})."""
    T = np.asarray(T).astype(int).ravel()
    if not set(np.unique(T).tolist()).issubset({0, 1}):
        raise ValueError("propensity_binary requires binary T in {0, 1}.")
    return propensity_matrix(X, T, 2, clip=clip, model=model)[:, 1]
