"""DoublyRobust-O-X (CAPPED) — robust AIPW value maximiser over the O (box) set, WITH a capacity constraint.

The doubly-robust (AIPW) objective of ``DoublyRobust`` applied to IPW-O-X's uncertainty set: an outcome model
μ̂_k(X) (the "direct method") plus the IPW correction, making ONLY the residual correction robust over the
marginal-sensitivity box (Rosenbaum Γ) ∩ per-arm Hájek calibration — the SAME (O) set as IPW-O-X:

    max_π   (1/n) Σ_i Σ_k π_k(X_i) μ̂_k(X_i)                                     [direct, outcome model]
          + min_{w ∈ box(Γ) ∩ per-arm Hájek calib.} (1/n) Σ_i π_{T_i}(X_i) w_i ( Y_i − μ̂_{T_i}(X_i) )   [robust residual]
    s.t.  Σ_k π_k(X_i)=1, π≥0, same-cell tying, (1/n)Σ_i π_k(X_i) ≤ cap_k.

This is the AIPW analog of IPW-O-X (and is identical to the box-only ``DoublyRobust`` method). Setting μ̂≡0 makes
the residual = Y and the direct term = 0, so it reduces EXACTLY to IPW-O-X (the correctness check). The full
problem is in ``code 1.1/methods/DoublyRobust-O-X/DoublyRobust-O-X.html``.

Just the method: NO statistical preprocessing — both the nominal inverse weights ``ips_weights`` and the
outcome means ``outcome_means`` are estimated upstream by ``common`` (NEVER the true propensities/means of
the DGP). The only thing it controls is the covariate GEOMETRY/discretisation via ``discretize``/``mesh``.
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
from common.sensitivity import marginal_sensitivity_box       # the Γ box [a_i, b_i]
from common.gurobi_env import configure_gurobi_license        # solver licence


@dataclass
class RODoublyRobustResult:
    """Optimal robust-AIPW policy + the box dual certificate (box-only)."""
    objective_value: float                 # robust DR value at the optimum (direct + worst-case residual)
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    Gamma: float                           # sensitivity level the box was built at
    cap: Tuple[float, ...]                 # per-arm capacity enforced
    usage: np.ndarray                      # (K,) realised per-arm usage (≤ cap)
    beta: np.ndarray                       # (K,) per-arm Hájek-calibration dual (FREE sign)
    mu: np.ndarray                         # (n,) box lower-bound dual (≥ 0)
    nu: np.ndarray                         # (n,) box upper-bound dual (≥ 0)
    solver_status: int


def solve_doublyrobust_o_x_capped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    outcome_means: np.ndarray,
    *,
    n_arms: int,
    Gamma: float,
    cap: Sequence[float],
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> RODoublyRobustResult:
    """Solve the CAPPED DoublyRobust-O-X (box-only AIPW) LP and return the optimal policy + duals.

    ``outcome_means`` is the (n, K) nuisance μ̂_k(X_i) (e.g. ``common.outcome_means(...)``). Arguments
    otherwise mirror the capped IPW-O-X solver. ``Gamma=1`` collapses the box and it reduces to the
    (non-robust) AIPW value maximiser; ``outcome_means≡0`` reduces it exactly to IPW-O-X.
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
    if Gamma < 1.0:
        raise ValueError("Gamma must be >= 1.0.")
    cap = tuple(float(c) for c in cap)
    if len(cap) != K:
        raise ValueError(f"cap must have length n_arms={K}.")

    # residual on the factual arm: r_i = Y_i − μ̂_{T_i}(X_i) — the part the IPW correction must explain
    resid = Y - muhat[np.arange(n), T]

    # ---- 1. GEOMETRY / DISCRETISATION (the `discretize` switch) ----------------------------------
    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    groups = tie_groups(support_X, rounding_digits)          # tie π across same-cell units

    # ---- 2. MARGINAL-SENSITIVITY BOX on the residual correction ---------------------------------
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)

    # ---- 3. BUILD THE DUAL LP (IPW-O-X's box-only dual, but Y→resid, plus the direct outcome term) ---
    configure_gurobi_license()
    m = gp.Model("DoublyRobust-O-X-capped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    # --- decision variables ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # the policy π_k(X_i) ∈ [0,1]
    mu = m.addVars(n, lb=0.0, name="mu")                     # box lower-bound dual μ_i ≥ 0
    nu = m.addVars(n, lb=0.0, name="nu")                     # box upper-bound dual ν_i ≥ 0
    beta = m.addVars(K, lb=-GRB.INFINITY, name="beta")      # per-arm Hájek-calibration dual (FREE sign)

    # --- objective: DIRECT outcome term + robust residual-correction dual (maximise) ---
    obj = gp.LinExpr()
    for i in range(n):
        for k in range(K):
            obj += (1.0 / n) * float(muhat[i, k]) * pi[k, i]   # direct: (1/n)Σ_i Σ_k π_k μ̂_k(X_i)
    for i in range(n):
        obj += float(a_box[i]) * mu[i] - float(b_box[i]) * nu[i]   # box interval contribution (residual)
    for k in range(K):
        obj += float(n) * beta[k]                            # per-arm calibration contribution
    m.setObjective(obj, GRB.MAXIMIZE)

    # --- stationarity constraint per unit (couples π·residual to the box + calibration duals) ---
    # μ_i − ν_i == (1/n) resid_i π_{T_i}(X_i) − β_{T_i}   (IPW-O-X's constraint with Y replaced by resid)
    for i in range(n):
        t = int(T[i])
        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(resid[i]) * pi[t, i] - beta[t], name=f"stat_{i}")

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
    for k in range(K):                                        # *** CAPACITY ***: (1/n)Σ_i π_k ≤ cap_k
        m.addConstr((1.0 / n) * gp.quicksum(pi[k, i] for i in range(n)) <= cap[k], name=f"cap_{k}")

    # ---- 4. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"DoublyRobust-O-X-capped non-optimal (status={m.Status}), Gamma={Gamma}.")

    # ---- 5. EXTRACT -----------------------------------------------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)                              # realised per-arm usage (≤ cap)
    return RODoublyRobustResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        Gamma=float(Gamma), cap=cap, usage=usage,
        beta=np.array([beta[k].X for k in range(K)]),
        mu=np.array([mu[i].X for i in range(n)]), nu=np.array([nu[i].X for i in range(n)]),
        solver_status=int(m.Status),
    )
