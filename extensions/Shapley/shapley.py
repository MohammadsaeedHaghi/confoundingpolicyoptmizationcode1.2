"""Shapley off-support extension — code 1.1 / extensions.

A METHOD learns a policy only on its (discretised) support. To DEPLOY that policy at new covariates
(e.g. a held-out test set, or a dense grid for plotting), we extend it off-support. The Shapley
extension uses the closed-form Lipschitz min–max operator (PDF Lemma 1):

    π^S(x) = min_k max_j A_jk(x),     A_jk(x) = [ d_k π̂(x_j) + d_j π̂(x_k) ] / (d_j + d_k),
             d_i = ||x − x_i||,        A_kk(x) = π̂(x_k).

It is the tightest 1-Lipschitz interpolant consistent with the support values, and it is EXACT at the
support points (an exact-match fast path returns the stored value there). Ported from
``srpo.extension.{shapley, multiarm}`` (PDF/closed-form variant; the optional Gurobi "LP" variant is
omitted here — the closed form is the production path).

Public API
----------
    extract_support(X_train, pi_train)                 -> (support_X, support_pi)  unique-cell collapse
    shapley_pdf(x_new, support_X, support_pi)          -> scalar / (m,)            one-arm operator
    extend_with_shapley(X_test, support_X, support_pi) -> (m,)                     one-arm extension
    extend_with_shapley_multiarm(X_test, X_train, pi)  -> (K, m)                   K-arm (renormalised)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ["extract_support", "shapley_pdf", "extend_with_shapley", "extend_with_shapley_multiarm"]


def extract_support(X_train: np.ndarray, pi_train: np.ndarray, *, rounding_digits: int = 6
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse ``(X_train, pi_train)`` to unique (rounded) covariate cells.

    For each distinct covariate the support value is the MEAN of pi over the matching rows (a no-op
    when the policy was tied per cell, which it is for a discretised free-π method). Returns
    ``(support_X, support_pi)``.
    """
    X_train = np.asarray(X_train, dtype=float)
    pi_train = np.asarray(pi_train, dtype=float).ravel()
    if X_train.shape[0] != pi_train.shape[0]:
        raise ValueError("X_train and pi_train must have equal first dimension.")
    df = pd.DataFrame(np.round(X_train, rounding_digits))
    df["_pi"] = pi_train
    cols = list(df.columns[:-1])
    grouped = df.groupby(cols, sort=True, as_index=False)["_pi"].mean()
    return grouped.loc[:, cols].to_numpy(), grouped["_pi"].to_numpy()


def _shapley_pdf_one(x_new: np.ndarray, support_X: np.ndarray, support_pi: np.ndarray) -> float:
    """Single-point evaluation of ``π^S(x) = min_k max_j A_jk(x)``."""
    diff = support_X - x_new[None, :]
    d = np.sqrt((diff * diff).sum(axis=1))                    # (K,) distances to each support point
    K = d.shape[0]
    if (d <= 0.0).any():                                      # exact match ⇒ return that support value
        return float(support_pi[int(np.argmin(d))])
    sum_dd = d[:, None] + d[None, :]                          # (K, K)  d_j + d_k
    safe = np.where(sum_dd > 0, sum_dd, 1.0)                  # avoid /0 on the diagonal (handled next)
    A = (d[None, :] / safe) * support_pi[:, None] \
      + (d[:, None] / safe) * support_pi[None, :]             # A_jk(x)
    diag = np.arange(K)
    A[diag, diag] = support_pi                                # A_kk(x) = π̂(x_k)
    return float(np.min(np.max(A, axis=0)))                   # min_k max_j A_jk


def shapley_pdf(x_new, support_X: np.ndarray, support_pi: np.ndarray):
    """Closed-form Shapley operator for a single covariate ``(d,)`` or a batch ``(m, d)``."""
    support_X = np.asarray(support_X, dtype=float)
    support_pi = np.asarray(support_pi, dtype=float).ravel()
    x_new = np.asarray(x_new, dtype=float)
    if x_new.ndim == 1 and x_new.shape[0] == support_X.shape[1]:
        return _shapley_pdf_one(x_new, support_X, support_pi)
    if x_new.ndim == 2 and x_new.shape[1] == support_X.shape[1]:
        return np.array([_shapley_pdf_one(x, support_X, support_pi) for x in x_new])
    raise ValueError(f"x_new has shape {x_new.shape}; expected (d,) or (m, d).")


def extend_with_shapley(X_test: np.ndarray, support_X: np.ndarray, support_pi: np.ndarray) -> np.ndarray:
    """One-arm Shapley extension of ``support_pi`` to every row of ``X_test``."""
    return np.atleast_1d(np.asarray(shapley_pdf(X_test, support_X, support_pi), dtype=float))


def _normalise(P: np.ndarray) -> np.ndarray:
    """Columns (one per test point) → simplex; uniform fallback where the column sums to ~0."""
    P = np.clip(np.asarray(P, dtype=float), 0.0, None)
    s = P.sum(axis=0, keepdims=True)
    return np.where(s > 1e-12, P / np.where(s > 1e-12, s, 1.0), 1.0 / P.shape[0])


def extend_with_shapley_multiarm(X_test: np.ndarray, X_train: np.ndarray, pi_train: np.ndarray) -> np.ndarray:
    """Extend a ``(K, n_train)`` on-support policy to ``X_test`` per arm, renormalised to a simplex.

    Returns ``(K, n_test)``. Each arm's probability function is extended independently by the scalar
    Shapley operator, then the K values at every test point are renormalised so Σ_k π_k(x) = 1. At the
    support points Shapley is exact and the renormalisation is a no-op.
    """
    pi_train = np.asarray(pi_train, dtype=float)              # (K, n_train)
    K = pi_train.shape[0]
    cols = []
    for k in range(K):
        sX, sp = extract_support(X_train, pi_train[k])        # unique cells + per-cell value for arm k
        cols.append(extend_with_shapley(X_test, sX, sp).ravel())
    return _normalise(np.vstack(cols))                       # (K, n_test) on the simplex
