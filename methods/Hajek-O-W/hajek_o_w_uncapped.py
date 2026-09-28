"""Hajek-O-W (UNCAPPED) — robust, Odds-box ∩ Wasserstein worst-case-REGRET minimiser, NO capacity.

Identical to the capped Hajek-O-W (``methods/Hajek-O-W/hajek_o_w_capped.py``) except that
the per-arm capacity constraint  (1/n) Σ_i π_k(X_i) ≤ cap_k  is **removed**: here the policy may
assign any arm to any fraction of the population. Every other part — the worst-case-REGRET
objective over the Marginal-Sensitivity box ∩ Wasserstein balls (a MIN LP, ≤ 0 because the
all-control baseline π₀ is feasible ⇒ do-no-harm), the simplex, the tie (discretisation)
constraints, and the factual/transport dual constraints — is unchanged. This is formerly
``srpo.multiarm.solve_kallus_w_multiarm``; the in-sample problem is written out in
``code 1.1/methods/Hajek-O-W/Hajek-O-W.html``.

Like the capped version this file is JUST the method: no statistical preprocessing (the nominal
inverse weights ``ips_weights`` are estimated upstream by ``common``, NEVER the true
propensities); the only thing it controls is the covariate GEOMETRY/discretisation via the
``discretize`` and ``mesh`` arguments.
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
class HajekOWResult:
    """Optimal Hajek-O-W policy + dual certificate (no capacity here, so no cap field)."""
    objective_value: float                 # worst-case REGRET at the optimum (≤ 0; do-no-harm)
    pi: np.ndarray                         # (K, n) optimal policy on the support; columns sum to 1
    support_X: np.ndarray                  # (n, d) the support each unit sits on (snapped or raw X)
    n_arms: int
    Gamma: float                           # sensitivity level the box was built at
    epsilon: Tuple[float, ...]             # per-arm Wasserstein radius actually used
    usage: np.ndarray                      # (K,) realised per-arm usage (1/n)Σ_i π_k  (UNCONSTRAINED)
    beta: np.ndarray                       # (K,) Wasserstein-radius dual (β_k ≥ 0)
    mu: np.ndarray                         # (n,) Kallus regret-box dual p_i (≥ 0)
    nu: np.ndarray                         # (n,) Kallus regret-box dual q_i (≥ 0)
    gamma_dual: Dict[int, np.ndarray]      # transport-"demand" dual γ_k (per arm, length n)
    theta: Dict[int, np.ndarray]           # transport-"factual" dual θ_k (per arm, NaN off-arm)
    solver_status: int


def solve_hajek_o_w_uncapped(
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
    zscore: bool = False,
    metric: str = "euclidean",
    lipschitz=None,
    lipschitz_k: int = 10,
    c_eps: float = 1.0,
    epsilon: Optional[Sequence[float]] = None,
    rounding_digits: int = 6,
    debug: bool = False,
) -> HajekOWResult:
    """Solve the UNCAPPED Hajek-O-W (MIN) dual LP and return the optimal policy + duals.

    Same arguments as the capped solver **minus** ``cap`` (there is no capacity constraint).
    See the capped file / ``code 1.1/methods/Hajek-O-W/Hajek-O-W.html`` for the full description of ``discretize``, ``mesh``,
    ``zscore``/``metric`` (the Wasserstein ground-cost geometry), ``c_eps`` and ``epsilon``.
    """
    # ---- 0. coerce inputs (no statistical preprocessing happens here) ----------------------------
    T = np.asarray(T).astype(int).ravel()                     # observed arm per unit
    Y = np.asarray(Y, dtype=float).ravel()                    # observed outcome per unit
    Yc = Y if maximize else -Y                                # reward (default) vs the paper's loss (negate Y)
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

    # ---- 2. WASSERSTEIN GEOMETRY: (optionally z-scored) pairwise distance matrix on the support ---
    if metric != "euclidean":
        raise ValueError(f"metric {metric!r} not supported (only 'euclidean').")
    D, _, _ = distance_matrix(support_X, zscore=zscore)           # (n, n) ground cost; z-scored if zscore=True

    # ---- 3. WASSERSTEIN RADII ε_k (tight per arm × c_eps), unless overridden ---------------------
    if epsilon is None:
        epsilon = tight_epsilon(D, T, w_hat, K, is_distance=True, c_eps=c_eps)
    epsilon = tuple(float(e) for e in epsilon)
    if len(epsilon) != K:
        raise ValueError(f"epsilon must have length n_arms={K}.")

    # ---- 4. MARGINAL-SENSITIVITY BOX: per-unit Γ interval [a_i, b_i] -----------------------------
    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)

    i_by_t = {k: np.where(T == k)[0].tolist() for k in range(K)}  # factual rows per arm
    for k in range(K):
        if not i_by_t[k]:
            raise ValueError(f"Treatment arm {k} has no observations.")

    # ---- 5. BUILD THE DUAL LP (same as capped, but NO capacity constraint) -----------------------
    configure_gurobi_license()
    m = gp.Model("Hajek-O-W-uncapped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    # --- decision variables (identical to the capped formulation) ---
    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")          # policy π_k(X_i) ∈ [0,1]
    beta = m.addVars(K, lb=0.0, name="beta")                 # Wasserstein-radius dual β_k ≥ 0
    gd = m.addVars(K, n, lb=-GRB.INFINITY, name="gd")        # transport-demand dual (free)
    theta = {(k, j): m.addVar(lb=-GRB.INFINITY, name=f"th_{k}_{j}")   # transport-factual dual (free)
             for k in range(K) for j in i_by_t[k]}
    p = m.addVars(n, lb=0.0, name="p")                       # Kallus regret-box dual p_i ≥ 0
    q = m.addVars(n, lb=0.0, name="q")                       # Kallus regret-box dual q_i ≥ 0

    # --- objective: worst-case REGRET (dual form), MINIMISE ---
    obj = gp.LinExpr()
    for k in range(K):
        obj += beta[k] * epsilon[k]                         # Wasserstein-radius cost
        obj += (1.0 / n) * gp.quicksum(gd[k, i] for i in range(n))   # transport-demand term
    for i in range(n):
        obj += float(b_box[i]) * p[i] - float(a_box[i]) * q[i]       # MSM-box interval term
    m.setObjective(obj, GRB.MINIMIZE)

    # --- policy constraints: simplex + tie ONLY (NO capacity) ---
    for i in range(n):                                        # each unit's policy is a distribution
        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f"simplex_{i}")
    for g in groups:                                         # tie policy across same-cell units
        a = int(g[0])
        for j in g[1:]:
            for k in range(K):
                m.addConstr(pi[k, a] == pi[k, int(j)], name=f"tie_{k}_{a}_{int(j)}")
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

    # *** NO capacity constraint here — that is the only structural difference from the capped LP ***

    # --- factual-row dual equality (regret reward c_{k,i}=(1/n)(1[k=0]−π_{k,i})Y_i) ---
    for k in range(K):
        for i in i_by_t[k]:
            rhs = (1.0 / n) * float(Yc[i]) if k == 0 else 0.0    # (1/n)·1[k=0]·Y_i
            m.addConstr(p[i] - q[i] - (1.0 / n) * theta[k, i] + (1.0 / n) * float(Yc[i]) * pi[k, i] == rhs,
                        name=f"feas_{k}_{i}")

    # --- transport-metric constraints (Wasserstein ball geometry) ---
    for k in range(K):
        for i in range(n):
            for j in i_by_t[k]:
                m.addConstr(beta[k] * float(D[i, j]) + gd[k, i] + theta[k, j] >= 0.0,
                            name=f"metric_{k}_{i}_{j}")

    # ---- 6. SOLVE -------------------------------------------------------------------------------
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Hajek-O-W-uncapped non-optimal (status={m.Status}), Gamma={Gamma}, eps={epsilon}.")

    # ---- 7. EXTRACT the optimal policy + dual certificate ---------------------------------------
    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)
    gd_out = {k: np.array([gd[k, i].X for i in range(n)]) for k in range(K)}
    th_out = {}
    for k in range(K):
        col = np.full(n, np.nan)
        for j in i_by_t[k]:
            col[j] = theta[k, j].X
        th_out[k] = col
    return HajekOWResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        Gamma=float(Gamma), epsilon=epsilon, usage=usage,
        beta=np.array([beta[k].X for k in range(K)]),
        mu=np.array([p[i].X for i in range(n)]), nu=np.array([q[i].X for i in range(n)]),
        gamma_dual=gd_out, theta=th_out, solver_status=int(m.Status),
    )
