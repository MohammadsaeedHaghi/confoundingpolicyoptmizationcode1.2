"""IPW-O-X (UNCAPPED) — robust, Odds-box-only value maximiser, WITHOUT a capacity constraint.

Identical to the capped IPW-O-X (``methods/IPW-O-X/ipw_o_x_capped.py``) except the per-arm capacity
constraint  (1/n) Σ_i π_k(X_i) ≤ cap_k  is **removed**. Everything else — the box-only worst-case
objective over the Marginal-Sensitivity box (+ per-arm Hájek calibration Σ W = n), the simplex, the
tie (discretisation) constraints, and the stationarity constraint — is unchanged. The in-sample
problem is written out in ``code 1.1/methods/IPW-O-X/IPW-O-X.html``.


IPW-O-X is IPW-O-W minus the Wasserstein term, so there is NO distance matrix and NO Wasserstein radius
(``c_eps``/``epsilon`` do not apply). Like every method here it does no statistical preprocessing
(weights are estimated upstream, NEVER the true propensities); it only controls the covariate
GEOMETRY/discretisation via ``discretize`` and ``mesh``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Make the code 1.2 root importable so ``common`` resolves.
# parents: [0]=method folder  [1]=methods  [2]=code 1.2
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.support import snap_to_grid, tie_groups           # discretisation (the "discretize" switch)
from common.sensitivity import marginal_sensitivity_box       # the Γ box [a_i, b_i]
from common.gurobi_env import configure_gurobi_license        # solver licence


@dataclass
class ROResult:
    """Optimal IPW-O-X policy + dual certificate (box-only; no capacity here, so no cap field)."""
    objective_value: float                 # worst-case IPW value at the optimum
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    Gamma: float                           # sensitivity level the box was built at
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (UNCONSTRAINED)
    beta: np.ndarray                       # (K,) per-arm Hájek-CALIBRATION dual (FREE sign — not a radius)
    mu: np.ndarray                         # (n,) box lower-bound dual (≥ 0)
    nu: np.ndarray                         # (n,) box upper-bound dual (≥ 0)
    solver_status: int


def solve_ipw_o_x_uncapped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    *,
    n_arms: int,
    Gamma: float,
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> ROResult:
    """Solve the UNCAPPED IPW-O-X dual LP and return the optimal policy + duals.

    Same arguments as the capped IPW-O-X solver **minus** ``cap`` (no capacity constraint).
    """
    # ---- 0. coerce inputs (no statistical preprocessing) ----------------------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    w_hat = np.asarray(ips_weights, dtype=float).ravel()      # nominal inverse weights (estimated upstream)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    if lipschitz is not None and K != 2:
        raise ValueError("lipschitz constrains only pi[1]: defined for n_arms=2 only.")
    if Gamma < 1.0:
        raise ValueError("Gamma must be >= 1.0.")

    # ---- 1. GEOMETRY / DISCRETISATION (the `discretize` switch) ------------------------------------
    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X   # snap to grid, or use raw X
    groups = tie_groups(support_X, rounding_digits)               # tie π across same-cell units

    # ---- 2. MARGINAL-SENSITIVITY BOX: per-unit Γ interval [a_i, b_i] -----------------------------
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)

    # ---- 3. BUILD THE DUAL LP (box-only; same as capped but NO capacity constraint) --------------
    configure_gurobi_license()
    m = gp.Model("IPW-O-X-uncapped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    # --- decision variables (identical to the capped formulation) ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # policy π_k(X_i) ∈ [0,1]
    mu = m.addVars(n, lb=0.0, name="mu")                     # box lower dual μ_i ≥ 0
    nu = m.addVars(n, lb=0.0, name="nu")                     # box upper dual ν_i ≥ 0
    beta = m.addVars(K, lb=-GRB.INFINITY, name="beta")      # per-arm Hájek-calibration dual (free sign)

    # --- objective: worst-case IPW value (dual form), maximise ---
    obj = gp.LinExpr()
    for i in range(n):
        obj += float(a_box[i]) * mu[i] - float(b_box[i]) * nu[i]   # MSM-box interval term
    for k in range(K):
        obj += float(n) * beta[k]                            # per-arm calibration term

    m.setObjective(obj, GRB.MAXIMIZE)

    # --- stationarity constraint per unit ---
    for i in range(n):
        t = int(T[i])
        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(Y[i]) * pi[t, i] - beta[t],
                    name=f"stat_{i}")

    # --- policy constraints: simplex + tie ONLY (NO capacity) ---
    if lipschitz is not None:                              # L-Lipschitz policy class
        _Xs = np.asarray(support_X, float)
        if _Xs.ndim == 1:
            _Xs = _Xs.reshape(-1, 1)
        if _Xs.shape[1] == 1:                             # 1-D: consecutive sorted pairs (exact)
            _o = np.argsort(_Xs.ravel()); _xs = _Xs.ravel()[_o]
            for _a in range(n - 1):
                _i, _j = int(_o[_a]), int(_o[_a + 1]); _dx = float(_xs[_a + 1] - _xs[_a])
                m.addConstr(pi[1, _i] - pi[1, _j] <= lipschitz * _dx)
                m.addConstr(pi[1, _j] - pi[1, _i] <= lipschitz * _dx)
        else:                                             # d > 1: k-NN pairs only (a relaxation of
            _DL = np.sqrt(((_Xs[:, None, :] - _Xs[None, :, :]) ** 2).sum(-1))   # the all-pairs condition)
            _kk = int(min(max(1, lipschitz_k), n - 1))
            for _i in range(n):
                for _jj in np.argsort(_DL[_i])[1:_kk + 1]:
                    _j = int(_jj)
                    if _j <= _i:
                        continue
                    _dx = float(_DL[_i, _j])
                    m.addConstr(pi[1, _i] - pi[1, _j] <= lipschitz * _dx)
                    m.addConstr(pi[1, _j] - pi[1, _i] <= lipschitz * _dx)

    for i in range(n):                                        # each unit's policy is a distribution
        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f"simplex_{i}")
    for g in groups:                                         # tie policy across same-cell units
        a = int(g[0])
        for j in g[1:]:
            for k in range(K):
                m.addConstr(pi[k, a] == pi[k, int(j)], name=f"tie_{k}_{a}_{int(j)}")
    # *** NO capacity constraint — the only structural difference from the capped LP ***

    # ---- 4. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"IPW-O-X-uncapped non-optimal (status={m.Status}), Gamma={Gamma}.")

    # ---- 5. EXTRACT the optimal policy + dual certificate ---------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)
    return ROResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        Gamma=float(Gamma), usage=usage,
        beta=np.array([beta[k].X for k in range(K)]),
        mu=np.array([mu[i].X for i in range(n)]), nu=np.array([nu[i].X for i in range(n)]),
        solver_status=int(m.Status),
    )
