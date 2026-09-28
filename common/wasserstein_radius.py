"""Tight per-arm Wasserstein radius ε — code 1.1 / common.

The Wasserstein covariate-balance ball in the O-W methods (IPW-O-W, DoublyRobust-O-W, Hajek-O-W) has a radius ε_k per arm. The
*tightest feasible* radius is the minimum transport cost that maps the empirical covariate law
to the arm-k inverse-weight-reweighted law — i.e. the smallest ε for which the balance
constraint is non-vacuous. Methods then use ``c_eps · ε_tight`` (c_eps = 1.0 is the robust
default). Ported from ``srpo.multiarm.good_epsilon_arm`` / ``tight_epsilon_multiarm``.

Needs Gurobi (it is itself a small transport LP), so it lives next to the solver infra.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .gurobi_env import configure_gurobi_license
from .geometry import pairwise_distance_matrix

__all__ = ["tight_epsilon_arm", "tight_epsilon"]


def tight_epsilon_arm(D: np.ndarray, T: np.ndarray, w_hat: np.ndarray, arm: int, *, debug: bool = False) -> float:
    """Minimum transport cost mapping the empirical X-law to the arm-``arm`` reweighted law.

    Solves:  min Σ_{i, j∈I_t} D_ij ζ_ij
             s.t. Σ_i ζ_ij = w_j / n  (j ∈ I_t),   Σ_j ζ_ij = 1/n  (∀ i),   ζ ≥ 0.
    ``D`` is the (n×n) covariate distance matrix; ``w_hat`` are the per-arm-normalised inverse
    weights (so each arm's weights sum to n).
    """
    configure_gurobi_license()                                # ensure a licence is active
    n = D.shape[0]
    I_arm = np.where(np.asarray(T).astype(int) == arm)[0].tolist()  # factual units of this arm
    if not I_arm:
        raise ValueError(f"Treatment arm {arm} is empty.")

    m = gp.Model("tight_epsilon_arm")
    m.Params.OutputFlag = 1 if debug else 0
    # transport plan ζ_ij ≥ 0: mass moved from every unit i to each arm-arm factual unit j
    zeta = {(i, j): m.addVar(lb=0.0, name=f"z_{i}_{j}") for i in range(n) for j in I_arm}
    # objective: total transport cost Σ D_ij ζ_ij (the Wasserstein cost)
    m.setObjective(gp.quicksum(float(D[i, j]) * zeta[i, j] for i in range(n) for j in I_arm), GRB.MINIMIZE)
    # column marginals: mass arriving at arm-unit j must equal its reweighted share w_j/n
    for j in I_arm:
        m.addConstr(gp.quicksum(zeta[i, j] for i in range(n)) == float(w_hat[j]) / n)
    # row marginals: each unit i supplies exactly its empirical mass 1/n
    for i in range(n):
        m.addConstr(gp.quicksum(zeta[i, j] for j in I_arm) == 1.0 / n)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"tight_epsilon_arm non-optimal (status={m.Status}) arm={arm}.")
    return float(m.ObjVal)                                    # the tightest feasible radius for this arm


def tight_epsilon(
    X_or_D: np.ndarray,
    T: np.ndarray,
    w_hat: np.ndarray,
    n_arms: int,
    *,
    is_distance: bool = False,
    c_eps: float = 1.0,
) -> Tuple[float, ...]:
    """Length-K tuple of (scaled) tight Wasserstein radii, one per arm.

    Pass either the covariate support (``is_distance=False`` → a distance matrix is built) or a
    precomputed (n×n) distance matrix (``is_distance=True``). Each tight radius is multiplied by
    ``c_eps`` (1.0 = robust default).
    """
    D = np.asarray(X_or_D, float) if is_distance else pairwise_distance_matrix(X_or_D)
    return tuple(c_eps * tight_epsilon_arm(D, T, w_hat, k) for k in range(n_arms))
