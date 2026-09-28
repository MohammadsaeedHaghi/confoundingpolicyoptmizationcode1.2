"""DoublyRobust-X-X (CAPPED) — plain AIPW value maximiser, NO uncertainty set, WITH a capacity constraint.

The standard doubly-robust (augmented-IPW) value at the NOMINAL weights ŵ — no odds-box (the 2nd slot is
X), no Wasserstein (the 3rd slot is X). The weights are NOT ranged over any uncertainty set:

    max_π   (1/n) Σ_i [ Σ_k π_k(X_i) μ̂_k(X_i)  +  ŵ_i ( Y_i − μ̂_{T_i}(X_i) ) π_{T_i}(X_i) ]
    s.t.    Σ_k π_k(X_i)=1, π≥0, same-cell tying, (1/n)Σ_i π_k(X_i) ≤ cap_k.

Equivalently  V_AIPW(π) = (1/n) Σ_i Σ_k π_k(X_i)[ μ̂_k(X_i) + 1[T_i=k] ŵ_i (Y_i − μ̂_k(X_i)) ]  — the
direct outcome term plus the IPW correction of the residual. This is the **non-robust** baseline that the
box-only ``DoublyRobust-O-X`` and the box∩Wasserstein ``DoublyRobust-O-W`` reduce to at Γ=1 (their box
collapses to the nominal point ŵ). It is the "AIPW" baseline used in the experiments.

Just the method: NO statistical preprocessing — both ``ips_weights`` (ŵ, nominal) and ``outcome_means``
(μ̂) are estimated upstream by ``common`` (NEVER the true propensities/means). It only controls the
covariate GEOMETRY/discretisation via ``discretize``/``mesh``. There is no Γ, no box, no duals — a direct LP.
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
class DoublyRobustResult:
    """Optimal plain-AIPW policy (no uncertainty set, so no robustness duals)."""
    objective_value: float                 # AIPW value at the optimum (direct + nominal IPW residual correction)
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    cap: Tuple[float, ...]                 # per-arm capacity enforced
    usage: np.ndarray                      # (K,) realised per-arm usage (≤ cap)
    solver_status: int


def solve_doublyrobust_x_x_capped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    outcome_means: np.ndarray,
    *,
    n_arms: int,
    Gamma: float = 1.0,                     # accepted for signature parity with the DR family; IGNORED (no uncertainty set)
    cap: Sequence[float],
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> DoublyRobustResult:
    """Solve the CAPPED plain-AIPW value maximiser (no box, no Wasserstein) and return the optimal policy.

    ``outcome_means`` is the (n, K) nuisance μ̂_k(X_i); ``ips_weights`` the nominal ŵ. ``Gamma`` is accepted
    only so the call signature matches the robust DR variants — it is not used (there is no uncertainty set).
    """
    # ---- 0. coerce inputs (no statistical preprocessing) ----------------------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    w_hat = np.asarray(ips_weights, dtype=float).ravel()      # nominal inverse weights (estimated upstream)
    muhat = np.asarray(outcome_means, dtype=float)            # (n, K) outcome-model means μ̂_k(X_i)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    if lipschitz is not None and K != 2:
        raise ValueError("lipschitz constrains only pi[1]: defined for n_arms=2 only.")
    if muhat.shape != (n, K):
        raise ValueError(f"outcome_means must be (n,K)=({n},{K}), got {muhat.shape}.")
    cap = tuple(float(c) for c in cap)
    if len(cap) != K:
        raise ValueError(f"cap must have length n_arms={K}.")

    # residual on the factual arm: r_i = Y_i − μ̂_{T_i}(X_i); per-unit IPW correction coefficient ŵ_i r_i
    resid = Y - muhat[np.arange(n), T]
    corr = w_hat * resid                                      # the factual-arm correction the policy may "claim"

    # ---- 1. GEOMETRY / DISCRETISATION (the `discretize` switch) ----------------------------------
    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    groups = tie_groups(support_X, rounding_digits)          # tie π across same-cell units

    # ---- 2. BUILD THE LP (direct outcome term + NOMINAL IPW residual correction; a direct max) ---
    configure_gurobi_license()
    m = gp.Model("DoublyRobust-X-X-capped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # the policy π_k(X_i) ∈ [0,1]

    # objective: (1/n) Σ_i [ Σ_k μ̂_k(X_i) π_k(X_i) + ŵ_i r_i π_{T_i}(X_i) ]  — linear in π, no duals
    obj = gp.LinExpr()
    for i in range(n):
        t = int(T[i])
        for k in range(K):
            obj += (1.0 / n) * float(muhat[i, k]) * pi[k, i]   # direct outcome-model term (all arms)
        obj += (1.0 / n) * float(corr[i]) * pi[t, i]           # IPW correction on the factual arm only
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
        raise RuntimeError(f"DoublyRobust-X-X-capped non-optimal (status={m.Status}).")

    # ---- 4. EXTRACT -----------------------------------------------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)                              # realised per-arm usage (≤ cap)
    return DoublyRobustResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        cap=cap, usage=usage, solver_status=int(m.Status),
    )
