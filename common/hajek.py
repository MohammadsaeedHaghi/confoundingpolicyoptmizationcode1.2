"""Inverse-propensity weights and Hájek (self-normalising) calibration — code 1.1 / common.

From an estimated propensity matrix ``P`` (n, K) and observed treatments ``T`` we build the
per-unit **inverse factual weight** (Horvitz–Thompson)::

    w_i = 1 / P(T = T_i | X_i)

and optionally apply the **Hájek per-arm normalisation** that rescales the weights *within
each arm* so they sum to a target (the project default is to sum to the full sample size n)::

    w_i  <-  w_i * (target_k / Σ_{j : T_j = k} w_j)      for i in arm k

Why Hájek
---------
Raw HT weights are high-variance and uncalibrated (a single tiny propensity can dominate the
estimate). The self-normalised (Hájek) weights are the convention the downstream policy LPs
expect: they stabilise the value estimate and embed the empirical per-arm calibration at the
*input* stage, so the optimisers consume a clean, balanced weight vector.

Public API
----------
    hajek_normalize(w, T[, target])           -> (n,) per-arm-rescaled weights
    inverse_propensity_weights(P, T[, ...])   -> (n,) w_i (Hájek-normalised by default)
    ipw_weights_from_data(X, T[, ...])        -> (w, P) one-shot estimate + weights
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .propensity import propensity_matrix, factual_propensity

__all__ = ["hajek_normalize", "inverse_propensity_weights", "ipw_weights_from_data"]

_TARGETS = ("n", "size", "global")


def hajek_normalize(w: np.ndarray, T: np.ndarray, *, target: str = "n") -> np.ndarray:
    """Rescale weights so they sum to a target.

    target
    ------
    ``"n"``      each arm's weights sum to the FULL sample size ``n`` (project convention —
                 matches the calibration ``Σ_{I_t} w = n`` the LPs assume).
    ``"size"``   each arm's weights sum to its own count ``n_k`` (classic Hájek per-arm mean,
                 i.e. weights average to 1 within an arm).
    ``"global"`` all weights sum to ``n`` (a single global rescale, no per-arm balancing).
    """
    if target not in _TARGETS:
        raise ValueError(f"target must be one of {_TARGETS}, got {target!r}.")
    w = np.asarray(w, dtype=float).copy()
    T = np.asarray(T).astype(int).ravel()
    if w.shape != T.shape:
        raise ValueError("w and T must have the same length.")
    n = w.shape[0]

    if target == "global":
        s = w.sum()
        if s == 0.0:
            raise RuntimeError("All weights are zero; check the propensities.")
        return w * (n / s)

    out = np.empty_like(w)
    for arm in np.unique(T):
        idx = np.where(T == arm)[0]
        s = w[idx].sum()
        if s == 0.0:
            raise RuntimeError(f"All inverse weights in arm {int(arm)} are zero; check propensities.")
        target_mass = float(n) if target == "n" else float(idx.size)
        out[idx] = w[idx] * (target_mass / s)
    return out


def inverse_propensity_weights(
    P: np.ndarray, T: np.ndarray, *, normalize: bool = True, target: str = "n"
) -> np.ndarray:
    """Per-unit inverse factual weight ``w_i = 1 / P(T=T_i | X_i)``.

    With ``normalize=True`` (default) the Hájek per-arm normalisation (``target``) is applied;
    with ``normalize=False`` the raw Horvitz–Thompson weights are returned.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    factual = factual_propensity(P, T)
    if np.any(factual <= 0.0):
        raise RuntimeError("Non-positive factual propensity; clip P before weighting.")
    w = 1.0 / factual
    return hajek_normalize(w, T, target=target) if normalize else w


def ipw_weights_from_data(
    X: np.ndarray,
    T: np.ndarray,
    n_arms: Optional[int] = None,
    *,
    clip: float = 1e-3,
    normalize: bool = True,
    target: str = "n",
) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot pre-algorithm step: estimate the propensity, then build the (Hájek-normalised)
    inverse weights. Returns ``(w, P)`` where ``w`` is ``(n,)`` and ``P`` is ``(n, K)`` — keep
    both (``P`` is needed for the sensitivity box and for provenance per the saving spec).
    """
    P = propensity_matrix(X, T, n_arms, clip=clip)
    w = inverse_propensity_weights(P, T, normalize=normalize, target=target)
    return w, P
