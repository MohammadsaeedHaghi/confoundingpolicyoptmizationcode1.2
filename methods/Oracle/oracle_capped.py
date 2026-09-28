"""Oracle (CAPPED) — best in-class policy with full value access, under a capacity constraint.

The Oracle is a **reference ceiling**, not a deployable method: it is handed a value matrix
``values`` (shape (n, K)) and picks the capacity-feasible policy that maximises the total value
on the (discretised) support:

        max_π  (1/n) Σ_i Σ_k π_k(X_i) · values[i, k]
        s.t.   Σ_k π_k(X_i)=1, π≥0, same-cell tying, (1/n) Σ_i π_k(X_i) ≤ cap_k.

Ported from ``srpo.multiarm.solve_oracle_capacity``. The full problem is in ``code 1.1/methods/Oracle/Oracle.html``.

Reference-only note: unlike the learned methods, the Oracle is *allowed* to use ground-truth value
information — pass ``values = Ypot`` (the per-unit potential outcomes ⇒ the clairvoyant "Full-info"
ceiling) or ``values = mu`` (the true outcome means μ_k(X) ⇒ the deployable "Best (true means)"
ceiling). It is never deployed as a policy; it only bounds how well any method could do. This does NOT
violate the no-true-propensities rule (which is about *learned* methods weighting with e_true).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Make the code 1.2 root importable so ``common`` resolves.
# parents: [0]=method folder  [1]=methods  [2]=code 1.2
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.support import snap_to_grid, tie_groups           # discretisation (the "discretize" switch)
from common.gurobi_env import configure_gurobi_license        # solver licence


@dataclass
class OracleResult:
    """Optimal capacity-constrained oracle policy on the supplied value matrix."""
    objective_value: float                 # the oracle value (1/n) Σ_i Σ_k π_k values[i,k]
    pi: np.ndarray                         # (K, n) oracle policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    cap: Tuple[float, ...]                 # per-arm capacity enforced
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (≤ cap)
    solver_status: int


def solve_oracle_capped(
    X: np.ndarray,
    values: np.ndarray,
    *,
    n_arms: int,
    cap: Sequence[float],
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
) -> OracleResult:
    """Solve the CAPPED oracle LP on ``values`` (n, K) and return the optimal policy.

    ``values[i, k]`` is the value of assigning arm ``k`` to unit ``i`` (e.g. Ypot for Full-info, or
    μ_k(X_i) for Best-true-means). ``discretize``/``mesh`` discretise X exactly as the learned methods,
    so the Oracle is a *fair* ceiling on the same support.
    """
    # ---- 0. coerce inputs -----------------------------------------------------------------------
    X = np.asarray(X, dtype=float)
    V = np.asarray(values, dtype=float)                       # (n, K) value matrix the oracle maximises
    n, K = V.shape[0], int(n_arms)
    if X.ndim != 2:
        X = X.reshape(n, -1)
    if V.shape[1] != K:
        raise ValueError(f"values must have n_arms={K} columns, got {V.shape[1]}.")
    cap = tuple(float(c) for c in cap)
    if len(cap) != K:
        raise ValueError(f"cap must have length n_arms={K}.")

    # ---- 1. GEOMETRY / DISCRETISATION (same support as the learned methods) ----------------------
    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X   # snap to grid, or use raw X
    groups = tie_groups(support_X, rounding_digits)               # tie π across same-cell units

    # ---- 2. BUILD THE ORACLE LP -----------------------------------------------------------------
    configure_gurobi_license()
    m = gp.Model("Oracle-capped")
    m.Params.OutputFlag = 1 if debug else 0

    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # the oracle policy π_k(X_i) ∈ [0,1]
    # objective: maximise the average assigned value (1/n) Σ_i Σ_k π_k(X_i) values[i,k]
    m.setObjective((1.0 / n) * gp.quicksum(float(V[i, k]) * pi[k, i]
                                           for k in range(K) for i in range(n)), GRB.MAXIMIZE)

    # --- policy constraints: simplex + tie + CAPACITY ---
    for i in range(n):                                        # each unit's policy is a distribution
        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f"simplex_{i}")
    for g in groups:                                         # tie policy across same-cell units
        a = int(g[0])
        for j in g[1:]:
            for k in range(K):
                m.addConstr(pi[k, a] == pi[k, int(j)], name=f"tie_{k}_{a}_{int(j)}")
    for k in range(K):                                        # *** CAPACITY ***: (1/n)Σ_i π_k ≤ cap_k
        m.addConstr((1.0 / n) * gp.quicksum(pi[k, i] for i in range(n)) <= cap[k], name=f"cap_{k}")

    # ---- 3. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Oracle-capped non-optimal (status={m.Status}).")

    # ---- 4. EXTRACT -----------------------------------------------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)                              # realised per-arm usage (≤ cap)
    return OracleResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        cap=cap, usage=usage, solver_status=int(m.Status),
    )
