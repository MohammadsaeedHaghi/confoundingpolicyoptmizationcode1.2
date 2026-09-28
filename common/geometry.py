"""Covariate geometry — code 1.1 / common.

The ground cost the Wasserstein covariate-balance term (the "W" in the O-W methods) is
built on: a pairwise distance matrix over the covariates, plus the standardisation that makes
those distances scale-fair.

Why it is a pre-algorithm step: the distance matrix ``D`` is a fixed function of ``X`` (it does
not depend on Γ or the policy), so it is computed once and reused by every Wasserstein method.
Distances are scale-sensitive, so standardise first (a feature in large units would otherwise
dominate the metric).

Public API
----------
    standardize(X[, mean, std])              -> (Xz, mean, std)   z-score; reuse train mean/std on test
    euclidean_distance(xi, xj)               -> float
    pairwise_distance_matrix(X)              -> (n, n) Euclidean distances
    distance_matrix(X, *, zscore=True)       -> (n, n) (optionally z-scored first) + the (mean, std) used
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "standardize",
    "euclidean_distance",
    "pairwise_distance_matrix",
    "distance_matrix",
]


def standardize(
    X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the columns of ``X``. Pass ``mean``/``std`` (fit on train) to apply on test.

    Constant columns (``std == 0``) are centred but not scaled (divided by 1), so they pass
    through without producing NaNs. Returns ``(Xz, mean, std)``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=0)
    safe_std = np.where(std > 0, std, 1.0)
    return (X - mean) / safe_std, np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def euclidean_distance(xi: np.ndarray, xj: np.ndarray) -> float:
    """Euclidean distance between two covariate points."""
    return float(np.linalg.norm(np.asarray(xi, dtype=float) - np.asarray(xj, dtype=float)))


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix, shape ``(n, n)`` (symmetric, zero diagonal)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def distance_matrix(
    X: np.ndarray,
    *,
    zscore: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Convenience: (optionally z-score then) build the pairwise distance matrix.

    Returns ``(D, mean, std)`` — keep ``(mean, std)`` so the same standardisation can be reused
    on a test set. With ``zscore=False`` the raw-X distances are returned and ``(mean, std)``
    are ``None``.
    """
    if not zscore:
        return pairwise_distance_matrix(X), None, None
    Xz, mu, sd = standardize(X, mean, std)
    return pairwise_distance_matrix(Xz), mu, sd
