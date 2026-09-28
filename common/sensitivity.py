"""Marginal Sensitivity Model — code 1.2 / common.

The unobserved confounder S makes the **estimated** inverse weights wrong by an unknown amount.
The Marginal Sensitivity Model bounds that error by a single factor Γ: the true (S-aware) weight
lies in a per-unit interval around the nominal weight. The nominal propensities are estimated and
the weights Hájek-normalised FIRST (``common.hajek``); the Γ box is then imposed inside the
optimisation problem around those nominal weights. The robust methods (IPW-O-W, IPW-O-X,
DoublyRobust-O-W, DoublyRobust-O-X, Hajek-O-X, Hajek-O-W) optimise the **worst case over this box**.

Γ* is the DGP's confounding level: the smallest Γ whose odds-ratio constraint holds for every
unit, i.e. the tight upper bound  max_{x,u} max(OR, 1/OR),  OR = odds(e(x,u)) / odds(e(x)).

Public API
----------
    gamma_star(e_xu, e_x)                    -> Γ*  (tight odds-ratio bound of a known DGP)
    marginal_sensitivity_box(w_hat, Gamma)   -> (a, b)  per-unit interval [a_i, b_i], a <= b
    augmented_gamma_grid(gammas, Gamma_star) -> sorted Γ-sweep with Γ* inserted
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

__all__ = ["gamma_star", "marginal_sensitivity_box", "augmented_gamma_grid"]


def gamma_star(e_xu, e_x) -> float:
    r"""Γ* — the tight upper bound on the odds ratio between the true and the nominal propensity.

    ``e_xu`` holds the true propensities P(T=1 | x, u) and ``e_x`` the nominal P(T=1 | x) = E_u[e(x,u)];
    they must broadcast against each other (e.g. ``e_xu`` of shape (m, |U|) and ``e_x`` of shape (m, 1)).
    Returns  max over all entries of  max(OR, 1/OR),  OR = [e_xu/(1-e_xu)] / [e_x/(1-e_x)].

    Because  w - 1 = (1-e)/e = 1/odds(e),  OR in [1/Γ, Γ] is exactly the condition that the true
    weight lies in the box  1 + t(ŵ - 1), t in [1/Γ, Γ];  the control arm gives 1/OR, which the
    symmetric max already covers. So Γ* is the smallest Γ whose box contains every true weight.
    """
    e_xu = np.asarray(e_xu, dtype=float)
    e_x = np.asarray(e_x, dtype=float)
    if np.any((e_xu <= 0) | (e_xu >= 1)) or np.any((e_x <= 0) | (e_x >= 1)):
        raise ValueError("propensities must lie strictly inside (0, 1).")
    log_or = np.log(e_xu / (1.0 - e_xu)) - np.log(e_x / (1.0 - e_x))
    return float(np.exp(np.max(np.abs(log_or))))


def marginal_sensitivity_box(w_hat: np.ndarray, Gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""Per-unit MSM interval on the confounded inverse weight.

    The Tan/Rosenbaum interval is :math:`\{1 + t\,(\hat w_i - 1) : t \in [\Gamma^{-1}, \Gamma]\}`,
    with endpoints :math:`1 + \Gamma^{-1}(\hat w_i - 1)` and :math:`1 + \Gamma(\hat w_i - 1)`.

    For :math:`\hat w_i \ge 1` the first endpoint is the lower one, but the per-arm Hájek
    normalisation can push some weights **below 1**, in which case the endpoints swap. Taking the
    element-wise ``min``/``max`` returns ``[a, b]`` with ``a <= b`` in either regime — without
    this a single ``w_hat`` just under 1 inverts its box and makes the dual LP unbounded (seen at
    strong confounding, γ_true = 5).

    Requires ``Gamma >= 1``. ``Gamma == 1`` collapses the box to a point (a = b = w_hat) — the
    no-robustness case where the robust methods coincide with IPW-X-X.
    """
    if Gamma < 1.0:
        raise ValueError("Gamma must be >= 1.0.")
    w_hat = np.asarray(w_hat, dtype=float)
    lo = 1.0 + (1.0 / Gamma) * (w_hat - 1.0)
    hi = 1.0 + Gamma * (w_hat - 1.0)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def augmented_gamma_grid(gammas: Iterable[float], Gamma_star: float, *, digits: int = 4) -> list:
    """Sorted Γ-sweep that always includes Γ* (see ``gamma_star``), so Γ* is a sampled point."""
    grid = {round(float(g), digits) for g in gammas}
    grid.add(round(float(Gamma_star), digits))
    return sorted(grid)
