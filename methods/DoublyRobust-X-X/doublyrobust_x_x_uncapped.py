"""DoublyRobust-X-X (UNCAPPED) — plain AIPW value maximiser, NO uncertainty set, no capacity.

Identical to the capped DoublyRobust-X-X (``methods/DoublyRobust-X-X/doublyrobust_x_x_capped.py``)
except the per-arm capacity constraint is removed. The standard doubly-robust (augmented-IPW) value at the
NOMINAL weights ŵ — no odds-box, no Wasserstein, no Γ:

    max_π   (1/n) Σ_i [ Σ_k π_k(X_i) μ̂_k(X_i)  +  ŵ_i ( Y_i − μ̂_{T_i}(X_i) ) π_{T_i}(X_i) ]
    s.t.    Σ_k π_k(X_i)=1, π≥0, same-cell tying.

Both ``ips_weights`` (ŵ) and ``outcome_means`` (μ̂) are estimated upstream (NEVER the true propensities/means).
A direct LP — no duals.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    """Optimal plain-AIPW policy (no uncertainty set, no capacity)."""
    objective_value: float
    pi: np.ndarray
    support_X: np.ndarray
    n_arms: int
    usage: np.ndarray
    solver_status: int


def solve_doublyrobust_x_x_uncapped(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    ips_weights: np.ndarray,
    outcome_means: np.ndarray,
    *,
    n_arms: int,
    Gamma: float = 1.0,                     # accepted for signature parity with the DR family; IGNORED
    discretize: bool = True,
    mesh: int = 6,
    mesh_range: Tuple[float, float] = (-1.0, 1.0),
    lipschitz=None,
    lipschitz_k: int = 10,
    rounding_digits: int = 6,
    debug: bool = False,
) -> DoublyRobustResult:
    """Solve the UNCAPPED plain-AIPW value maximiser (no box, no Wasserstein). ``Gamma`` is ignored."""
    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    w_hat = np.asarray(ips_weights, dtype=float).ravel()
    muhat = np.asarray(outcome_means, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(len(T), -1)
    n, K = X.shape[0], int(n_arms)
    if lipschitz is not None and K != 2:
        raise ValueError("lipschitz constrains only pi[1]: defined for n_arms=2 only.")
    if muhat.shape != (n, K):
        raise ValueError(f"outcome_means must be (n,K)=({n},{K}), got {muhat.shape}.")

    resid = Y - muhat[np.arange(n), T]
    corr = w_hat * resid

    lo, hi = mesh_range
    support_X = snap_to_grid(X, mesh, lo, hi) if discretize else X
    groups = tie_groups(support_X, rounding_digits)

    configure_gurobi_license()
    m = gp.Model("DoublyRobust-X-X-uncapped")
    m.Params.OutputFlag = 1 if debug else 0
    m.Params.DualReductions = 0

    pi = m.addVars(K, n, lb=0.0, ub=1.0, name="pi")
    obj = gp.LinExpr()
    for i in range(n):
        t = int(T[i])
        for k in range(K):
            obj += (1.0 / n) * float(muhat[i, k]) * pi[k, i]
        obj += (1.0 / n) * float(corr[i]) * pi[t, i]
    m.setObjective(obj, GRB.MAXIMIZE)

    for i in range(n):
        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f"simplex_{i}")
    for g in groups:
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
    # *** NO capacity constraint — the only structural difference from the capped LP ***

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"DoublyRobust-X-X-uncapped non-optimal (status={m.Status}).")

    pi_arr = np.clip(np.array([[pi[k, i].X for i in range(n)] for k in range(K)]), 0.0, 1.0)
    usage = pi_arr.mean(axis=1)
    return DoublyRobustResult(
        objective_value=float(m.ObjVal), pi=pi_arr, support_X=support_X, n_arms=K,
        usage=usage, solver_status=int(m.Status),
    )
