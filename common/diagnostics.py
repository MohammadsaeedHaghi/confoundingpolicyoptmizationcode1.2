"""Pre-algorithm diagnostics — code 1.1 / common.

Sanity checks on the ESTIMATED propensities / inverse weights BEFORE trusting any policy result.
Because the project never uses true propensities (see [[no_true_propensities]]), every method rides on
ê(X)=P(T=k|X) estimated from data: if covariate overlap is poor the inverse weights explode and the
"robust" worst-case value/regret becomes an artifact of a handful of units. These diagnostics surface
that — low effective sample size or a near-zero estimated propensity is the red flag.

Everything is computed from (X, T) only (propensities are ESTIMATED via common.propensity) — never from
e_true / the unobserved confounder S.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .propensity import propensity_matrix

__all__ = ["effective_sample_size", "propensity_diagnostics"]


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size of a weight vector:  (Σ w)² / Σ w².

    Equals the count when weights are uniform, and shrinks toward 1 as a few weights dominate.
    """
    w = np.asarray(w, dtype=float)
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def propensity_diagnostics(X, T, n_arms, *, min_overlap: float = 0.05) -> Dict:
    """Estimate ê(X) and report overlap / inverse-weight concentration, per arm and overall.

    Parameters
    ----------
    X, T        : covariates (n, d) and observed arm (n,). ``n_arms`` = number of arms K.
    min_overlap : the smallest acceptable estimated factual propensity; below it ``overlap_ok`` is False.

    Returns a JSON-able dict:
      n, n_arms, min_overlap,
      min_propensity      — global smallest factual ê_{T_i}(X_i) (overlap red-flag if tiny),
      max_weight          — largest raw inverse weight 1/ê (pre-Hájek),
      ess, ess_frac       — global Kish ESS and ESS/n (concentration of the inverse weights),
      overlap_ok          — min_propensity ≥ min_overlap,
      per_arm[k]          — {n, min_propensity, mean_propensity, max_weight, ess, ess_frac}.
    A low ess_frac (≪ 1) or overlap_ok=False ⇒ the weights are dominated by a few units and any
    downstream value/regret should be treated with suspicion.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T).astype(int).ravel()
    K, n = int(n_arms), len(T)
    P = propensity_matrix(X, T, K)                      # (n, K) ESTIMATED propensities, rows sum to 1
    e_fac = P[np.arange(n), T]                          # factual propensity ê_{T_i}(X_i)
    w = 1.0 / np.clip(e_fac, 1e-12, None)              # raw inverse weights (pre-Hájek)

    per_arm: Dict[int, Dict] = {}
    for k in range(K):
        idx = np.where(T == k)[0]
        if len(idx) == 0:
            per_arm[k] = dict(n=0, min_propensity=float("nan"), mean_propensity=float("nan"),
                              max_weight=float("nan"), ess=0.0, ess_frac=float("nan"))
            continue
        ek, wk = e_fac[idx], w[idx]
        ess_k = effective_sample_size(wk)
        per_arm[k] = dict(n=int(len(idx)), min_propensity=float(ek.min()),
                          mean_propensity=float(ek.mean()), max_weight=float(wk.max()),
                          ess=float(ess_k), ess_frac=float(ess_k / len(idx)))

    ess = effective_sample_size(w)
    min_prop = float(e_fac.min())
    return dict(
        n=n, n_arms=K, min_overlap=float(min_overlap),
        min_propensity=min_prop, max_weight=float(w.max()),
        ess=float(ess), ess_frac=float(ess / n),
        overlap_ok=bool(min_prop >= min_overlap),
        per_arm=per_arm,
    )
