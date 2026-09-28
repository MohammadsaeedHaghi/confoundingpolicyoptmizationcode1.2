"""IPW-O-W (CAPPED) — robust, Odds-box ∩ Wasserstein value maximiser, WITH a capacity constraint.

This is the user's RW method (formerly ``srpo.multiarm.solve_rw_multiarm``) in the clean
clean form. It learns the policy π that maximises the WORST-CASE inverse-propensity-weighted
value on the training sample, where the worst case is taken over the uncertainty set

        U(Γ, ε)  =  { Marginal-Sensitivity box (Rosenbaum Γ) }  ∩  { per-arm Wasserstein balls (ε_k) }

subject to π being a valid policy AND the per-arm capacity  (1/n) Σ_i π_k(X_i) ≤ cap_k.
The full in-sample problem (primal worst-case value + the dual LP solved here) is written out in
``code 1.1/methods/IPW-O-W/IPW-O-W.html``.

This file is JUST the method:
  * it does NO statistical preprocessing — the nominal inverse weights ``ips_weights`` are
    estimated upstream by ``common`` (NEVER the true propensities);
  * the ONE thing it controls is the GEOMETRY/discretisation of the covariates, via the
    ``discretize`` (snap-to-grid or not) and ``mesh`` (grid resolution) arguments.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Make the code 1.2 root importable so ``common`` resolves.
# parents: [0]=method folder  [1]=methods  [2]=code 1.2
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.support import snap_to_grid, tie_groups           # discretisation (the "discretize" switch)
from common.geometry import distance_matrix                  # Wasserstein ground cost D (z-scoreable)
from common.sensitivity import marginal_sensitivity_box       # the Γ box [a_i, b_i]
from common.wasserstein_radius import tight_epsilon           # tight per-arm radius ε_k
from common.gurobi_env import configure_gurobi_license        # solver licence


@dataclass
class ROWResult:
    """Optimal IPW-O-W policy + the dual certificate (everything the saving spec wants)."""
    objective_value: float                 # worst-case IPW value at the optimum
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    Gamma: float                           # sensitivity level the box was built at
    epsilon: Tuple[float, ...]             # per-arm Wasserstein radius actually used
    cap: Tuple[float, ...]                 # per-arm capacity enforced
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (≤ cap)
    beta: np.ndarray                       # (K,) Wasserstein-radius dual (β_k ≥ 0)
    mu: np.ndarray                         # (n,) box lower-bound dual (≥ 0)
    nu: np.ndarray                         # (n,) box upper-bound dual (≥ 0)
    gamma_dual: Dict[int, np.ndarray]      # transport-"demand" dual γ_k (per arm, length n)
    theta: Dict[int, np.ndarray]           # transport-"factual" dual θ_k (per arm, NaN off-arm)
    solver_status: int


def solve_ipw_o_w_capped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    *,
    n_arms: int,
    Gamma: float,
    cap: Sequence[float],
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    zscore: bool = False,
    metric: str = "euclidean",
    c_eps: float = 1.0,
    epsilon: Optional[Sequence[float]] = None,
    rounding_digits: int = 6,
    debug: bool = False,
    lipschitz=None,
    lipschitz_k: int = 10,
) -> ROWResult:
    """Solve the CAPPED IPW-O-W dual LP and return the optimal policy + duals.

    Parameters
    ----------
    X, T, Y      : covariates (n, d), observed arm (n,), observed outcome (n,).
    ips_weights  : nominal inverse weights ŵ_i (estimated upstream — NEVER true propensities).
    n_arms       : number of arms K (arm 0 = control by convention).
    Gamma        : Rosenbaum/MSM sensitivity (Γ ≥ 1). Γ=1 ⇒ no robustness.
    cap          : per-arm capacity (length K); the constraint (1/n)Σ_i π_k ≤ cap_k is enforced.
    discretize   : if True, DISCRETISE X onto a grid (snap, so the free-π LP is non-degenerate);
                   if False, use the raw X (singleton support — the degenerate / no-grid case).
    mesh         : number of grid levels per coordinate when discretize=True (e.g. 6 ⇒ 6×6 cells in 2-D).
    mesh_range   : (lo, hi) span the grid covers per coordinate.
    zscore       : z-score (standardise) covariates before building the Wasserstein ground cost D, so a
                   large-unit feature cannot dominate the metric. Default False = raw-X distances
                   (bit-identical to the original solver); the config path defaults this to True.
    metric       : ground-cost metric for D (only "euclidean" is implemented).
    c_eps        : scale on the tight Wasserstein radius (1.0 = robust default).
    epsilon      : override the per-arm Wasserstein radius directly (skips the tight-ε solve).
    rounding_digits : precision at which support points are considered "the same" cell (tie grouping).
    """
    # ---- 0. coerce inputs (no statistical preprocessing happens here) ----------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    w_hat = np.asarray(ips_weights, dtype=float).ravel()      # nominal inverse weights (estimated upstream)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:                                           # accept 1-D X as a single column
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    if lipschitz is not None and K != 2:
        raise ValueError("lipschitz constrains only pi[1]: defined for n_arms=2 only.")
    if Gamma < 1.0:
        raise ValueError("Gamma must be >= 1.0.")
    cap = tuple(float(c) for c in cap)
    if len(cap) != K:
        raise ValueError(f"cap must have length n_arms={K}.")

    # ---- 1. GEOMETRY / DISCRETISATION (the `discretize` switch) ------------------------------------
    lo, hi = mesh_range
    # discretize=True  -> snap each unit onto the mesh grid so many units share a cell (support.py);
    # discretize=False -> keep raw X (every unit its own cell => singleton/degenerate support).
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    # Tie groups: units that landed on the SAME support point get one tied policy => π is a
    # genuine function of X (piecewise-constant on cells). On raw X every group has size 1.
    groups = tie_groups(support_X, rounding_digits)

    # ---- 2. WASSERSTEIN GEOMETRY: (optionally z-scored) pairwise distance matrix on the support ---
    if metric != "euclidean":
        raise ValueError(f"metric {metric!r} not supported (only 'euclidean').")
    D, _, _ = distance_matrix(support_X, zscore=zscore)      # (n, n) ground cost; z-scored if zscore=True

    # ---- 3. WASSERSTEIN RADII ε_k (tight per arm × c_eps), unless explicitly overridden ----------
    if epsilon is None:
        epsilon = tight_epsilon(D, T, w_hat, K, is_distance=True, c_eps=c_eps)
    epsilon = tuple(float(e) for e in epsilon)
    if len(epsilon) != K:
        raise ValueError(f"epsilon must have length n_arms={K}.")

    # ---- 4. MARGINAL-SENSITIVITY BOX: per-unit Γ interval [a_i, b_i] on the true weight ----------
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)

    # Which units were observed under each arm (the "factual" rows for that arm).
    i_by_t = {k: np.where(T == k)[0].tolist() for k in range(K)}
    for k in range(K):
        if not i_by_t[k]:
            raise ValueError(f"Treatment arm {k} has no observations.")

    # ---- 5. BUILD THE DUAL LP (the worst-case-value dual; see ``code 1.1/methods/IPW-O-W/IPW-O-W.html``) -------------------------
    configure_gurobi_license()
    m = gp.Model("IPW-O-W-capped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0                              # keep the dual solution well-defined

    # --- decision variables ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # the policy π_k(X_i) ∈ [0,1]
    beta = m.addVars(K, lb=0.0, name="beta")                 # Wasserstein-radius dual β_k ≥ 0
    gd = m.addVars(K, n, lb=-GRB.INFINITY, name="gd")        # transport-demand dual γ_{k,i} (free)
    theta = {(k, j): m.addVar(lb=-GRB.INFINITY, name=f"th_{k}_{j}")   # transport-factual dual θ_{k,j} (free)
             for k in range(K) for j in i_by_t[k]}
    mu = m.addVars(n, lb=0.0, name="mu")                     # box lower-bound dual μ_i ≥ 0
    nu = m.addVars(n, lb=0.0, name="nu")                     # box upper-bound dual ν_i ≥ 0

    # --- objective: the worst-case IPW value in dual form (maximise) ---
    obj = gp.LinExpr()
    for k in range(K):
        obj += -beta[k] * epsilon[k]                        # cost of the Wasserstein radius ε_k
        obj += (1.0 / n) * gp.quicksum(gd[k, i] for i in range(n))   # transport-demand contribution
    for i in range(n):
        obj += mu[i] * float(a_box[i]) - nu[i] * float(b_box[i])     # MSM-box interval contribution
    m.setObjective(obj, GRB.MAXIMIZE)

    # --- policy constraints: simplex, tie, CAPACITY ---
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
    for g in groups:                                         # tie the policy across same-cell units
        a = int(g[0])
        for j in g[1:]:
            for k in range(K):
                m.addConstr(pi[k, a] == pi[k, int(j)], name=f"tie_{k}_{a}_{int(j)}")
    for k in range(K):                                        # *** CAPACITY ***: (1/n)Σ_i π_k ≤ cap_k
        m.addConstr((1.0 / n) * gp.quicksum(pi[k, i] for i in range(n)) <= cap[k], name=f"cap_{k}")

    # --- dual feasibility on factual rows (links π·Y to the box + transport duals) ---
    for k in range(K):
        for i in i_by_t[k]:
            m.addConstr((1.0 / n) * pi[k, i] * float(Y[i]) + (1.0 / n) * theta[k, i]
                        - mu[i] + nu[i] >= 0.0,
                        name=f"feas_{k}_{i}")

    # --- transport-metric constraints (the Wasserstein ball geometry, using D) ---
    for k in range(K):
        for i in range(n):
            for j in i_by_t[k]:
                m.addConstr(beta[k] * float(D[i, j]) - gd[k, i] - theta[k, j] >= 0.0,
                            name=f"metric_{k}_{i}_{j}")

    # ---- 6. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"IPW-O-W-capped non-optimal (status={m.Status}), Gamma={Gamma}, eps={epsilon}.")

    # ---- 7. EXTRACT the optimal policy + dual certificate ---------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)                              # realised per-arm usage (≤ cap)
    gd_out = {k: np.array([gd[k, i].X for i in range(n)]) for k in range(K)}
    th_out = {}
    for k in range(K):                                       # θ is defined only on each arm's factual rows
        col = np.full(n, np.nan)
        for j in i_by_t[k]:
            col[j] = theta[k, j].X
        th_out[k] = col
    return ROWResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        Gamma=float(Gamma), epsilon=epsilon, cap=cap, usage=usage,
        beta=np.array([beta[k].X for k in range(K)]),
        mu=np.array([mu[i].X for i in range(n)]), nu=np.array([nu[i].X for i in range(n)]),
        gamma_dual=gd_out, theta=th_out, solver_status=int(m.Status),
    )
