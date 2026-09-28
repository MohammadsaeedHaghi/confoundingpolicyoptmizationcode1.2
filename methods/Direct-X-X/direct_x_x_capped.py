"""Direct-X-X (CAPPED) — the NAIVE factual-reward maximiser, WITH a capacity constraint.

Direct-X-X is the WEAKEST baseline here (formerly ``srpo.multiarm.solve_direct_opt_multiarm``).
It learns the policy π that maximises the raw observed factual reward on the training sample with NO
inverse-propensity weighting (unit weights ŵ_i ≡ 1) and NO confounding adjustment whatsoever — it simply
trusts the confounded data as-is. Subject to π being a valid policy AND the per-arm capacity
(1/n) Σ_i π_k(X_i) ≤ cap_k. The full in-sample problem is written out in
``code 1.1/methods/Direct-X-X/Direct-X-X.html``.

Just the method: no statistical preprocessing, and — unlike every other method here — it does NOT even take
``ips_weights`` (it is IPW-X-X with all weights pinned to 1). The only thing it controls is the covariate
GEOMETRY/discretisation via ``discretize`` and ``mesh``. (Direct-X-X uses NO sensitivity box, NO
distance matrix and NO Wasserstein radius — so there is no ``Gamma``/``c_eps``/``epsilon`` here. The
discretisation is still needed: it pools units into cells so the free-π LP is non-degenerate.)
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
from common.gurobi_env import configure_gurobi_license        # solver licence


@dataclass
class DirectResult:
    """Optimal Direct-X-X policy (naive primal LP: no sensitivity / no transport duals)."""
    objective_value: float                 # naive factual IPW value (w≡1) at the optimum
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    cap: Tuple[float, ...]                 # per-arm capacity enforced
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (≤ cap)
    solver_status: int


def solve_direct_x_x_capped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    *,
    n_arms: int,
    cap: Sequence[float],
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> DirectResult:
    """Solve the CAPPED Direct-X-X LP and return the optimal policy.

    Parameters mirror the capped IPW-O-X solver, **minus** every robustness control: there is no
    ``ips_weights`` (weights are pinned to 1), no ``Gamma`` (no sensitivity box), and no Wasserstein
    controls. This is exactly IPW-X-X with ŵ ≡ 1 — the most naive, fully data-trusting baseline.
    """
    # ---- 0. coerce inputs (no statistical preprocessing) ----------------------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    if lipschitz is not None and K != 2:
        raise ValueError("lipschitz constrains only pi[1]: defined for n_arms=2 only.")
    cap = tuple(float(c) for c in cap)
    if len(cap) != K:
        raise ValueError(f"cap must have length n_arms={K}.")

    # ---- 1. GEOMETRY / DISCRETISATION (the `discretize` switch) ------------------------------------
    lo, hi = mesh_range
    # discretize=True -> snap onto a mesh grid so units pool into cells; discretize=False -> raw X.
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    groups = tie_groups(support_X, rounding_digits)          # tie π across same-cell units

    # ---- 2. BUILD THE NAIVE PRIMAL LP (raw factual reward; see ``code 1.1/methods/Direct-X-X/Direct-X-X.html``) ---------
    configure_gurobi_license()
    m = gp.Model("Direct-X-X-capped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    # --- decision variables ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # the policy π_k(X_i) ∈ [0,1]

    # --- objective: raw factual reward (IPW with w≡1), maximise ---
    # (1/n) Σ_i Y_i · π_{T_i}(X_i) — NO inverse weight, NO confounding correction.
    obj = gp.LinExpr()
    for i in range(n):
        obj += (1.0 / n) * float(Y[i]) * pi[int(T[i]), i]    # unit-weight factual reward of unit i
    m.setObjective(obj, GRB.MAXIMIZE)

    # --- policy constraints: simplex + tie + CAPACITY ---
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
        else:                                             # d > 1: k-NN pairs (a relaxation)
            _DL = np.sqrt(((_Xs[:, None, :] - _Xs[None, :, :]) ** 2).sum(-1))
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
    for k in range(K):                                        # *** CAPACITY ***: (1/n)Σ_i π_k ≤ cap_k
        m.addConstr((1.0 / n) * gp.quicksum(pi[k, i] for i in range(n)) <= cap[k], name=f"cap_{k}")

    # ---- 3. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Direct-X-X-capped non-optimal (status={m.Status}).")

    # ---- 4. EXTRACT the optimal policy ----------------------------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)                              # realised per-arm usage (≤ cap)
    return DirectResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        cap=cap, usage=usage, solver_status=int(m.Status),
    )
