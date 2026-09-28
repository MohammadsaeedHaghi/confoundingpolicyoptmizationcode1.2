"""Hajek-O-X (UNCAPPED) — robust, Odds-box-only REGRET minimiser, WITHOUT a capacity constraint.

Identical to the capped Hajek-O-X (``methods/Hajek-O-X/hajek_o_x_capped.py``) except the
per-arm capacity constraint  (1/n) Σ_i π_k(X_i) ≤ cap_k  is **removed**. Everything else — the
box-only worst-case REGRET objective over the Marginal-Sensitivity box (+ per-arm Hájek
calibration Σ W = n), the simplex, the tie (discretisation) constraints, and the per-unit regret
constraint — is unchanged. The in-sample problem is written out in
``code 1.1/methods/Hajek-O-X/Hajek-O-X.html``.

Hajek-O-X is the Kallus & Zhou regret-minimiser over the box only (IPW-O-W minus the Wasserstein
term), so there is NO distance matrix and NO Wasserstein radius (``c_eps``/``epsilon`` do not
apply). Like every method here it does no statistical preprocessing (weights are estimated
upstream, NEVER the true propensities); it only controls the covariate GEOMETRY/discretisation
via ``discretize`` and ``mesh``.
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
class HajekOResult:
    """Optimal Hajek-O-X policy + dual certificate (box-only; no capacity here, so no cap field)."""
    objective_value: float                 # worst-case REGRET at the optimum (≤ 0: do-no-harm)
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    Gamma: float                           # sensitivity level the box was built at
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (UNCONSTRAINED)
    beta: np.ndarray                       # (K,) per-treatment self-normalised worst-case regret level λ_t (Q̂_t)
    mu: np.ndarray                         # (n,) Charnes–Cooper lower-bound dual u_i (≥ 0)
    nu: np.ndarray                         # (n,) Charnes–Cooper upper-bound dual v_i (≥ 0)
    solver_status: int


def solve_hajek_o_x_uncapped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    *,
    n_arms: int,
    Gamma: float,
    maximize: bool = True,
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> HajekOResult:
    """Solve the UNCAPPED Hajek-O-X MIN LP and return the optimal policy + duals.

    Same arguments as the capped Hajek-O-X solver **minus** ``cap`` (no capacity constraint). The
    all-control policy π₀ is always feasible ⇒ the optimal worst-case regret is ≤ 0 (do-no-harm).
    """
    # ---- 0. coerce inputs (no statistical preprocessing) ----------------------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    Yc = Y if maximize else -Y                                # reward (default) vs the paper's loss (negate Y)
    # SELF-NORMALISED Hájek: box on RAW inverse weights W̃=1/ê ≥ 1 (a_i ≥ 1 > 0 keeps the per-treatment denom > 0).
    w_hat = np.asarray(ips_weights, dtype=float).ravel()      # nominal RAW inverse weights (estimated upstream)
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

    # ---- 2. MARGINAL-SENSITIVITY BOX: per-unit Γ interval [a_i, b_i] on the RAW weight -----------
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)
    if np.any(a_box <= 0.0):
        raise ValueError("Hajek-O-X self-normalised: a_i ≤ 0 — pass RAW inverse weights (≥1), not Hájek-rescaled.")
    i_by_t = {k: np.where(T == k)[0].tolist() for k in range(K)}

    # ---- 3. BUILD THE MIN LP (self-normalised Hájek worst-case regret; same as capped but NO capacity) --
    # Per treatment t:  Q̂_t = sup_{a≤W≤b} Σ_{I_t} r_i W_i / Σ_{I_t} W_i,  r_i = (1[t=0] − π_t(x_i)) Y_i,
    # via Charnes–Cooper duality (Kallus & Zhou Eq. 12):
    #   Q̂_t = min_{u,v≥0, λ_t} λ_t  s.t.  v_i − u_i + λ_t ≥ r_i ∀i∈I_t,  Σ_{i∈I_t}(u_i a_i − v_i b_i) ≥ 0.
    configure_gurobi_license()
    m = gp.Model("Hajek-O-uncapped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    # --- decision variables ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # policy π_k(X_i) ∈ [0,1]
    u = m.addVars(n, lb=0.0, name="u")                       # CC lower-bound dual u_i ≥ 0
    v = m.addVars(n, lb=0.0, name="v")                       # CC upper-bound dual v_i ≥ 0
    lam = m.addVars(K, lb=-GRB.INFINITY, name="lam")         # per-treatment worst-case regret level λ_t = Q̂_t

    # --- objective: total worst-case self-normalised regret Σ_t Q̂_t, MINIMISE ---
    m.setObjective(gp.quicksum(lam[k] for k in range(K)), GRB.MINIMIZE)

    # --- CC dual coupling:  v_i − u_i + λ_{T_i} + Y_i π_{T_i,i} ≥ 1[T_i=0] Y_i ---
    for i in range(n):
        t = int(T[i])
        rhs = float(Yc[i]) if t == 0 else 0.0
        m.addConstr(v[i] - u[i] + lam[t] + float(Yc[i]) * pi[t, i] >= rhs, name=f"cc_{i}")
    for k in range(K):                                        # CC box-product: Σ_{I_t}(u_i a_i − v_i b_i) ≥ 0
        m.addConstr(gp.quicksum(u[i] * float(a_box[i]) - v[i] * float(b_box[i]) for i in i_by_t[k]) >= 0.0,
                    name=f"ccbox_{k}")

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
        raise RuntimeError(f"Hajek-O-uncapped non-optimal (status={m.Status}), Gamma={Gamma}.")

    # ---- 5. EXTRACT the optimal policy + dual certificate ---------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)
    return HajekOResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        Gamma=float(Gamma), usage=usage,
        beta=np.array([lam[k].X for k in range(K)]),
        mu=np.array([u[i].X for i in range(n)]), nu=np.array([v[i].X for i in range(n)]),
        solver_status=int(m.Status),
    )
