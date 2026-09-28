"""KNN inverse-distance off-support extension — code 1.1 / extensions.

The simpler alternative to the Shapley operator for deploying an on-support policy at new covariates:
at each test point, average the support policy over the k nearest support points, weighted by inverse
distance. Ported from ``srpo.extension.{knn, multiarm}``.

    weights[j] = 1 / (||x − support_X[j]|| + eps),   j ∈ k-NN(x);   weights /= Σ weights
    π̂(x)      = Σ_j weights[j] · support_pi[j]

Public API
----------
    extend_with_knn(X_test, support_X, support_pi[, k, eps])   -> (m,)      one-arm extension
    extend_with_knn_multiarm(X_test, X_train, pi[, k, eps])    -> (K, m)    K-arm (renormalised)
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

# extract_support is shared with the Shapley extension (unique-cell collapse + per-cell mean).
import sys
from pathlib import Path
_SHAPLEY = Path(__file__).resolve().parents[1] / "Shapley"
if str(_SHAPLEY) not in sys.path:
    sys.path.insert(0, str(_SHAPLEY))
from shapley import extract_support  # noqa: E402

__all__ = ["extend_with_knn", "extend_with_knn_multiarm"]


def extend_with_knn(
    X_test: np.ndarray, support_X: np.ndarray, support_pi: np.ndarray, *, k: int = 50, eps: float = 1e-8
) -> np.ndarray:
    """Inverse-distance KNN average of ``support_pi`` at each ``X_test`` row (one arm).

    ``k`` is clamped to the number of support points. Output clipped to [0, 1].
    """
    support_X = np.asarray(support_X, dtype=float)
    support_pi = np.asarray(support_pi, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    if support_X.ndim != 2 or support_X.shape[0] != support_pi.shape[0]:
        raise ValueError("support_X must be (n_support, d) and match support_pi length.")
    if X_test.shape[1] != support_X.shape[1]:
        raise ValueError("X_test and support_X must share covariate dimension.")

    k_eff = min(int(k), support_pi.shape[0])                  # cannot ask for more neighbours than exist
    knn = NearestNeighbors(n_neighbors=k_eff).fit(support_X)
    dists, idxs = knn.kneighbors(X_test, n_neighbors=k_eff)   # (m, k_eff) distances + support indices
    weights = 1.0 / (dists + eps)                            # inverse-distance weights
    weights /= weights.sum(axis=1, keepdims=True)             # normalise per test point
    out = np.einsum("ij,ij->i", weights, support_pi[idxs])   # weighted average of support values
    return np.clip(out, 0.0, 1.0)


def _normalise(P: np.ndarray) -> np.ndarray:
    """Columns (one per test point) → simplex; uniform fallback where the column sums to ~0."""
    P = np.clip(np.asarray(P, dtype=float), 0.0, None)
    s = P.sum(axis=0, keepdims=True)
    return np.where(s > 1e-12, P / np.where(s > 1e-12, s, 1.0), 1.0 / P.shape[0])


def extend_with_knn_multiarm(
    X_test: np.ndarray, X_train: np.ndarray, pi_train: np.ndarray, *, k: int = 50, eps: float = 1e-8
) -> np.ndarray:
    """Extend a ``(K, n_train)`` on-support policy to ``X_test`` per arm via KNN, renormalised.

    Returns ``(K, n_test)``. Each arm extended independently, then the K values at every test point are
    renormalised to the simplex.
    """
    pi_train = np.asarray(pi_train, dtype=float)              # (K, n_train)
    K = pi_train.shape[0]
    cols = []
    for arm in range(K):
        sX, sp = extract_support(X_train, pi_train[arm])      # unique cells + per-cell value for this arm
        cols.append(extend_with_knn(X_test, sX, sp, k=k, eps=eps).ravel())
    return _normalise(np.vstack(cols))                       # (K, n_test) on the simplex
