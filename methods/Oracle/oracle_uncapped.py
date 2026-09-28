"""Oracle (UNCAPPED) — best in-class policy with full value access, NO capacity constraint.

Identical to the capped Oracle (``methods/Oracle/oracle_capped.py``) but with the per-arm
capacity constraint removed. With no capacity, the oracle reduces to the per-cell **argmax of the
average value** (each support cell is assigned, all-or-nothing, to its best-value arm):

        max_π  (1/n) Σ_i Σ_k π_k(X_i) · values[i, k]
        s.t.   Σ_k π_k(X_i)=1, π≥0, same-cell tying.

Like the capped version this is a **reference ceiling**, not a deployable method — it is handed the
value matrix ``values`` (Ypot for the clairvoyant Full-info ceiling, or μ for the Best-true-means
ceiling, or use it for the unconstrained "best arm, no cap" map). See ``code 1.1/methods/Oracle/Oracle.html``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    """Optimal oracle policy on the supplied value matrix (no capacity here, so no cap field)."""
    objective_value: float                 # the oracle value (1/n) Σ_i Σ_k π_k values[i,k]
    pi: np.ndarray                         # (K, n) oracle policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    usage: np.ndarray                      # (K,) realised per-arm usage (UNCONSTRAINED)
    solver_status: int


def solve_oracle_uncapped(
    X: np.ndarray,
    values: np.ndarray,
    *,
    n_arms: int,
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
) -> OracleResult:
    """Solve the UNCAPPED oracle LP on ``values`` (n, K). Same as the capped solver minus ``cap``."""
    # ---- 0. coerce inputs -----------------------------------------------------------------------
    X = np.asarray(X, dtype=float)
    V = np.asarray(values, dtype=float)                       # (n, K) value matrix
    n, K = V.shape[0], int(n_arms)
    if X.ndim != 2:
        X = X.reshape(n, -1)
    if V.shape[1] != K:
        raise ValueError(f"values must have n_arms={K} columns, got {V.shape[1]}.")

    # ---- 1. GEOMETRY / DISCRETISATION -----------------------------------------------------------
    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    groups = tie_groups(support_X, rounding_digits)

    # ---- 2. BUILD THE ORACLE LP (no capacity) ---------------------------------------------------
    configure_gurobi_license()
    m = gp.Model("Oracle-uncapped")
    m.Params.OutputFlag = 1 if debug else 0

    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # oracle policy π_k(X_i) ∈ [0,1]
    m.setObjective((1.0 / n) * gp.quicksum(float(V[i, k]) * pi[k, i]
                                           for k in range(K) for i in range(n)), GRB.MAXIMIZE)

    # --- policy constraints: simplex + tie ONLY (NO capacity) ---
    for i in range(n):
        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f"simplex_{i}")
    for g in groups:
        a = int(g[0])
        for j in g[1:]:
            for k in range(K):
                m.addConstr(pi[k, a] == pi[k, int(j)], name=f"tie_{k}_{a}_{int(j)}")
    # *** NO capacity constraint — the only structural difference from the capped Oracle ***

    # ---- 3. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Oracle-uncapped non-optimal (status={m.Status}).")

    # ---- 4. EXTRACT -----------------------------------------------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)
    return OracleResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        usage=usage, solver_status=int(m.Status),
    )
