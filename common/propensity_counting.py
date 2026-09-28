"""Counting (empirical-frequency) propensity estimation — code 1.1 / common.

For a **discrete** covariate ``X`` (finite support) the nominal propensity ``ê(X)=P(T=k|X)``
can be estimated by plain counting: group the units by their ``X`` cell and, within each cell,
read off the arm frequencies

    P(T = k | X = x) = #{i : X_i = x, T_i = k} / #{i : X_i = x}.

For binary treatment this is exactly "count the treated and the untreated at each X and divide".
This is the non-parametric, assumption-free counterpart to the logistic estimator in
``propensity.py``; it returns the **same** ``(n, K)`` clipped / row-normalised matrix, so it
drops straight into :func:`common.inverse_propensity_weights` and the policy optimisers.

Use it when ``X`` is discrete — counting is exact there and avoids the model mis-specification a
logistic fit can introduce. (On continuous ``X`` every unit is its own cell, so counting
degenerates; bin / snap ``X`` first — see ``common.support``.)

Convention (project-wide, identical to ``propensity.py``):
  * arms are indexed ``0 … K-1``; arm 0 is the control;
  * cell propensities are clipped to ``[clip, 1-clip]`` then row-renormalised so each row sums
    to 1 (a positivity / overlap safeguard so the inverse weights stay finite — a cell that
    never treats some arm gets ``clip`` there, not 0);
  * everything is ESTIMATED from ``(X, T)`` only — NEVER from ``e_true`` / the confounder ``S``.

Public API
----------
    count_propensity_table(X, T[, K])   -> PropensityTable   (per-cell counts + propensity)
    count_propensity_matrix(X, T[, K])  -> (n, K)  P(T=k | X_i) by counting
    count_propensity_binary(X, T)       -> (n,)    ê(X_i)=P(T=1 | X_i)   (K = 2 convenience)
    count_ipw_weights_from_data(X, T)   -> (w, P)  counting propensity + Hájek IPW weights
    CountingPropensity                  -> fit/predict (deploy a train propensity on new X)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .hajek import inverse_propensity_weights

__all__ = [
    "PropensityTable",
    "CountingPropensity",
    "count_propensity_table",
    "count_propensity_matrix",
    "count_propensity_binary",
    "count_ipw_weights_from_data",
]


def _as2d(X: np.ndarray, n: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(n, -1) if X.ndim != 2 else X


@dataclass
class PropensityTable:
    """The per-cell counting summary (what you get by 'looking at every value of X and counting')."""
    cells: np.ndarray          # (m, d)  the unique X cells (rounded)
    counts: np.ndarray         # (m, K)  arm counts in each cell  (#{T=k, X=cell})
    n_cell: np.ndarray         # (m,)    total units in each cell  (counts.sum(1))
    propensity: np.ndarray     # (m, K)  clipped + row-renormalised P(T=k | cell)
    marginal: np.ndarray       # (K,)    fallback P(T=k) for cells unseen at fit time
    n_arms: int
    rounding_digits: int

    def __repr__(self) -> str:                                   # compact, inspectable preview
        m, K = self.counts.shape
        head = []
        for r in range(min(m, 8)):
            xs = ", ".join(f"{v:g}" for v in self.cells[r])
            cs = " ".join(f"{int(c)}" for c in self.counts[r])
            ps = " ".join(f"{p:.3f}" for p in self.propensity[r])
            head.append(f"  X=({xs}): counts[{cs}] -> P[{ps}]")
        more = "" if m <= 8 else f"\n  … (+{m - 8} more cells)"
        return f"PropensityTable(K={K}, cells={m})\n" + "\n".join(head) + more


class CountingPropensity:
    """Empirical-frequency propensity estimator with a scikit-style ``fit`` / ``predict``.

    ``fit`` builds the per-cell arm-frequency table; ``predict`` looks each unit's ``X`` cell up
    and returns its arm probabilities (an unseen cell falls back to the training **marginal**
    ``P(T=k)``). Fitting on the training set and predicting on a held-out set therefore deploys
    the *train* propensity on new data — the analogue of reusing a fitted logistic ``model``.
    """

    def __init__(self, n_arms: Optional[int] = None, *, clip: float = 1e-3,
                 rounding_digits: int = 6, alpha: float = 0.0) -> None:
        self.n_arms = n_arms
        self.clip = float(clip)
        self.rounding_digits = int(rounding_digits)
        self.alpha = float(alpha)               # optional Laplace smoothing of the cell counts
        self.table_: Optional[PropensityTable] = None

    # -- internals ---------------------------------------------------------------------------
    def _clip_norm(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(P, self.clip, 1.0 - self.clip)
        return P / P.sum(axis=1, keepdims=True)

    def _keys(self, X: np.ndarray):
        return [tuple(row) for row in np.round(X, self.rounding_digits)]

    # -- API ---------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, T: np.ndarray) -> "CountingPropensity":
        T = np.asarray(T).astype(int).ravel()
        n = T.shape[0]
        X = _as2d(X, n)
        if np.any(T < 0):
            raise ValueError("treatment arms must be integers in 0 … K-1.")
        K = self.n_arms if self.n_arms is not None else int(T.max()) + 1
        if self.n_arms is None:
            self.n_arms = K

        keys = self._keys(X)
        index: dict = {}
        cells: list = []
        counts: list = []
        for key, t in zip(keys, T):
            if key not in index:
                index[key] = len(cells)
                cells.append(key)
                counts.append(np.zeros(K, dtype=float))
            counts[index[key]][t] += 1.0

        counts = np.asarray(counts)                       # (m, K)
        n_cell = counts.sum(axis=1)                       # (m,)
        prop_raw = (counts + self.alpha) / (n_cell[:, None] + self.alpha * K)
        propensity = self._clip_norm(prop_raw)
        marginal = self._clip_norm((np.bincount(T, minlength=K).astype(float) + self.alpha)[None, :])[0]

        self._index = index
        self.table_ = PropensityTable(
            cells=np.asarray(cells, dtype=float), counts=counts, n_cell=n_cell,
            propensity=propensity, marginal=marginal, n_arms=K, rounding_digits=self.rounding_digits,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the ``(n, K)`` propensity matrix for ``X`` (seen cells exact, unseen → marginal)."""
        if self.table_ is None:
            raise RuntimeError("call fit(X, T) before predict(X).")
        n = np.asarray(X).reshape(-1, 1).shape[0] if np.asarray(X).ndim == 1 else np.asarray(X).shape[0]
        X = _as2d(X, n)
        P = np.empty((n, self.n_arms), dtype=float)
        tab = self.table_
        for i, key in enumerate(self._keys(X)):
            j = self._index.get(key)
            P[i] = tab.propensity[j] if j is not None else tab.marginal
        return P

    def predict_factual(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """``P(T = T_i | X_i)`` — the observed-arm column of :meth:`predict`."""
        T = np.asarray(T).astype(int).ravel()
        P = self.predict(X)
        return P[np.arange(P.shape[0]), T]


# ----------------------------------------------------------------------------------------------
# functional wrappers (one-shot fit-and-read, mirroring propensity.py)
# ----------------------------------------------------------------------------------------------
def count_propensity_table(
    X: np.ndarray, T: np.ndarray, n_arms: Optional[int] = None, *,
    clip: float = 1e-3, rounding_digits: int = 6, alpha: float = 0.0,
) -> PropensityTable:
    """Per-cell counting summary: for each distinct ``X`` value, the arm counts and ``P(T=k|X)``."""
    return CountingPropensity(n_arms, clip=clip, rounding_digits=rounding_digits, alpha=alpha).fit(X, T).table_


def count_propensity_matrix(
    X: np.ndarray, T: np.ndarray, n_arms: Optional[int] = None, *,
    clip: float = 1e-3, rounding_digits: int = 6, alpha: float = 0.0,
) -> np.ndarray:
    """``(n, K)`` propensity matrix ``P(T=k | X_i)`` estimated by counting within ``X`` cells."""
    cp = CountingPropensity(n_arms, clip=clip, rounding_digits=rounding_digits, alpha=alpha).fit(X, T)
    return cp.predict(X)


def count_propensity_binary(
    X: np.ndarray, T: np.ndarray, *, clip: float = 1e-3, rounding_digits: int = 6, alpha: float = 0.0,
) -> np.ndarray:
    """Binary-treatment convenience: ``ê(X_i)=P(T=1 | X_i)`` by counting (``T ∈ {0,1}``)."""
    T = np.asarray(T).astype(int).ravel()
    if not set(np.unique(T).tolist()).issubset({0, 1}):
        raise ValueError("count_propensity_binary requires binary T in {0, 1}.")
    return count_propensity_matrix(X, T, 2, clip=clip, rounding_digits=rounding_digits, alpha=alpha)[:, 1]


def count_ipw_weights_from_data(
    X: np.ndarray, T: np.ndarray, n_arms: Optional[int] = None, *,
    clip: float = 1e-3, rounding_digits: int = 6, alpha: float = 0.0,
    normalize: bool = True, target: str = "n",
) -> Tuple[np.ndarray, np.ndarray]:
    """Counting analogue of :func:`common.ipw_weights_from_data`: estimate the propensity by
    counting, then build the (Hájek-normalised) inverse weights. Returns ``(w, P)``."""
    P = count_propensity_matrix(X, T, n_arms, clip=clip, rounding_digits=rounding_digits, alpha=alpha)
    w = inverse_propensity_weights(P, T, normalize=normalize, target=target)
    return w, P
