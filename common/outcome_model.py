"""Outcome-model nuisance μ̂_k(X) = E[Y | X, T=k] — code 1.1 / common.

The DoublyRobust method needs, for every arm k, a regression estimate of the conditional outcome
mean μ̂_k(X) — the "direct method" / outcome-model nuisance of the AIPW score. This is estimated from
the observed (X, T, Y) ONLY (never the true means, never e_true / S): for each arm k we fit a regressor
on that arm's factual rows and predict for ALL units (counterfactual prediction).

Public API
----------
    outcome_means(X, T, Y, n_arms[, cross_fit, kind, clip, seed])  -> (n, K)  μ̂_k(X_i)

``kind="auto"`` uses logistic regression when Y is binary (the studies' case ⇒ μ̂_k = P̂(Y=1|X,T=k))
and linear regression otherwise. ``cross_fit=True`` returns honest K-fold out-of-fold predictions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

__all__ = ["outcome_means"]


def _is_binary(Y: np.ndarray) -> bool:
    return set(np.unique(Y).tolist()).issubset({0.0, 1.0})


def _fit_predict_arm(Xtr, Ytr, Xpred, *, binary: bool):
    """Fit one arm's outcome regressor on (Xtr, Ytr) and predict at Xpred."""
    if Xtr.shape[0] == 0:                                     # no data for this arm in this split
        return np.full(Xpred.shape[0], float(np.mean(Ytr)) if Ytr.size else 0.0)
    if binary and len(np.unique(Ytr)) < 2:                   # degenerate (all 0 or all 1) ⇒ constant
        return np.full(Xpred.shape[0], float(Ytr.mean()))
    if binary:
        mdl = LogisticRegression(solver="lbfgs", max_iter=2000).fit(Xtr, Ytr.astype(int))
        return mdl.predict_proba(Xpred)[:, 1]
    return LinearRegression().fit(Xtr, Ytr).predict(Xpred)


def outcome_means(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_arms: int,
    *,
    cross_fit: bool = False,
    n_folds: int = 5,
    kind: str = "auto",
    clip: Optional[tuple] = None,
    seed: int = 0,
) -> np.ndarray:
    """Per-arm outcome means μ̂_k(X_i), shape (n, K).

    cross_fit=False : fit each arm's model on ALL its factual rows, predict every unit.
    cross_fit=True  : K-fold — for each held-out fold, fit each arm on the other folds' arm rows.
    clip            : optional (lo, hi) to clip predictions (e.g. (0,1) for a binary outcome).
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    binary = _is_binary(Y) if kind == "auto" else (kind == "logistic")
    mu = np.zeros((n, K), dtype=float)

    if not cross_fit:
        for k in range(K):                                   # fit on arm-k rows, predict all units
            idx = np.where(T == k)[0]
            mu[:, k] = _fit_predict_arm(X[idx], Y[idx], X, binary=binary)
    else:
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(n), n_folds)  # random K-fold split
        for f in folds:
            f = np.asarray(f, dtype=int)
            mask_tr = np.ones(n, bool); mask_tr[f] = False    # everything not in this fold = training
            for k in range(K):
                tr = np.where(mask_tr & (T == k))[0]          # other folds' arm-k rows
                mu[f, k] = _fit_predict_arm(X[tr], Y[tr], X[f], binary=binary)

    if clip is not None:
        mu = np.clip(mu, clip[0], clip[1])
    return mu
