"""Self-normalised (Hájek) worst-case regret — code 1.1 / common.

Kallus & Zhou's regret estimator is the PER-TREATMENT **self-normalised** Hájek ratio (Eq. 4, 10):

    R̂^{(t)}(π; W) = [ Σ_{i:T_i=t} (π(t|x_i) − π₀(t|x_i)) Y_i W_i ] / [ Σ_{i:T_i=t} W_i ],

i.e. the per-treatment denominator Σ_{T_i=t} W_i **floats with W** — a linear-FRACTIONAL program,
NOT a fixed Σ_{T_i=t}W=n calibration. The worst case distributes over treatments,
R̄(π;W^Γ) = Σ_t sup_{W∈W^Γ} R̂^{(t)}(π;W), each term solved here.

Sign convention. We use the REWARD convention (Y high = better): the per-unit regret coefficient is
``r_i = (π₀(T_i|x_i) − π(T_i|x_i)) Y_i = (1[T_i=0] − π_{T_i}(x_i)) Y_i`` (all-control baseline π₀).
Worst case = sup_W (adversary makes the policy look worst), and the methods MINIMISE Σ_t sup_W over π.
(Kallus assumes Y a loss; reward = −loss, so flip Y to recover the paper exactly.)

Box convention. The MSM box [a_i, b_i] must be built on the **raw** inverse weights W̃_i = 1/ê_{T_i}(x_i)
≥ 1 (so a_i ≥ 1 > 0 and the self-normalised denominator stays positive). Do NOT pass the per-arm
Hájek-rescaled weights here (those can dip below 1 and send a_i negative, blowing up the ratio).

Two independent solvers for the box case (cross-checked in tests):
  * ``selfnorm_box_dinkelbach`` — Dinkelbach root-find on the linear-fractional level (fast, no LP);
  * ``selfnorm_box_lp``         — the Charnes–Cooper LP (Eq. 11), returns the normalised w too.
And ``selfnorm_wasserstein_dinkelbach`` for the box ∩ per-arm Wasserstein ball (LP subproblem per λ).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from common.gurobi_env import configure_gurobi_license

__all__ = [
    "selfnorm_box_dinkelbach", "selfnorm_box_lp", "selfnorm_wasserstein_dinkelbach",
]


# --------------------------------------------------------------------------- box, Dinkelbach
def selfnorm_box_dinkelbach(r: np.ndarray, a: np.ndarray, b: np.ndarray,
                            *, tol: float = 1e-10, iters: int = 200) -> Tuple[float, np.ndarray]:
    """sup over W∈[a,b] of  (Σ r_i W_i)/(Σ W_i), via Dinkelbach bisection on the level λ.

    For a fixed λ, g(λ) = max_{W∈[a,b]} Σ (r_i − λ) W_i = Σ [ (r_i−λ) b_i if r_i>λ else (r_i−λ) a_i ],
    which is continuous and strictly decreasing in λ; the optimal ratio λ* is its unique root g(λ*)=0,
    and the maximiser is W_i = b_i where r_i>λ*, a_i where r_i<λ*. Requires a_i>0 (positive denom).
    """
    r = np.asarray(r, float); a = np.asarray(a, float); b = np.asarray(b, float)
    if r.size == 0:
        return 0.0, np.zeros(0)
    if np.any(a <= 0):
        raise ValueError("selfnorm box requires a_i > 0 (build the box on RAW inverse weights ≥ 1).")
    def g(lam):
        W = np.where(r > lam, b, a)
        return float(np.sum((r - lam) * W)), W
    lo, hi = float(r.min()), float(r.max())          # ratio lies within [min r, max r]
    if hi - lo < tol:
        W = b.copy(); return float(hi), W
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val, _ = g(mid)
        if val > 0:                                  # ratio achievable above mid
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    lam = 0.5 * (lo + hi)
    W = np.where(r > lam, b, a)
    ratio = float(np.sum(r * W) / np.sum(W))
    return ratio, W


# --------------------------------------------------------------------------- box, Charnes–Cooper LP
def selfnorm_box_lp(r: np.ndarray, a: np.ndarray, b: np.ndarray,
                    *, debug: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """Same sup as above via the Charnes–Cooper LP (Eq. 11):
        max_{w≥0, s≥0} Σ r_i w_i  s.t.  Σ w_i = 1,  s a_i ≤ w_i ≤ s b_i.
    Returns (value, W_star = w/s, w_normalised) — w sums to 1 over the treatment's units.
    """
    r = np.asarray(r, float); a = np.asarray(a, float); b = np.asarray(b, float)
    m = r.size
    if m == 0:
        return 0.0, np.zeros(0), np.zeros(0)
    configure_gurobi_license()
    md = gp.Model("selfnorm_cc")
    md.Params.OutputFlag = 1 if debug else 0
    w = md.addVars(m, lb=0.0, name="w")
    s = md.addVar(lb=0.0, name="s")
    md.addConstr(gp.quicksum(w[i] for i in range(m)) == 1.0, name="norm")
    for i in range(m):
        md.addConstr(w[i] <= s * float(b[i]), name=f"ub_{i}")
        md.addConstr(w[i] >= s * float(a[i]), name=f"lb_{i}")
    md.setObjective(gp.quicksum(float(r[i]) * w[i] for i in range(m)), GRB.MAXIMIZE)
    md.optimize()
    if md.Status != GRB.OPTIMAL:
        raise RuntimeError(f"selfnorm_box_lp non-optimal (status={md.Status}).")
    w_arr = np.array([w[i].X for i in range(m)])
    s_val = s.X
    W = w_arr / s_val if s_val > 1e-12 else w_arr.copy()
    return float(md.ObjVal), W, w_arr


# --------------------------------------------------------------------------- box ∩ Wasserstein, Dinkelbach
def selfnorm_wasserstein_dinkelbach(
    r_t: np.ndarray, a_t: np.ndarray, b_t: np.ndarray, D_t: np.ndarray, epsilon: float,
    *, tol: float = 1e-8, iters: int = 80, debug: bool = False,
) -> Tuple[float, np.ndarray]:
    """sup over W∈[a,b]∩Wasserstein-ball of (Σ r_j W_j)/(Σ W_j) for ONE treatment's arm-t units.

    Wasserstein ball: ∃ transport ζ_{ij} ≥ 0 (i over all n demand points, j over the m arm-t units)
    with Σ_j ζ_{ij}=1/n ∀i, Σ_i ζ_{ij}=W_j/n ∀j, Σ_{ij} D_{ij} ζ_{ij} ≤ ε. Dinkelbach on the level λ:
    for fixed λ, max_W Σ(r_j−λ)W_j over the box ∩ ball is an LP (W + ζ); bisect λ to the root.
    D_t is (n, m): distance from every demand point i to each arm-t unit j. Requires a_j>0.
    """
    r_t = np.asarray(r_t, float); a_t = np.asarray(a_t, float); b_t = np.asarray(b_t, float)
    D_t = np.asarray(D_t, float)
    m = r_t.size; n = D_t.shape[0]
    if m == 0:
        return 0.0, np.zeros(0)
    if np.any(a_t <= 0):
        raise ValueError("selfnorm Wasserstein requires a_j > 0 (box on RAW inverse weights ≥ 1).")
    configure_gurobi_license()

    def g(lam):
        md = gp.Model("sn_wass_level")
        md.Params.OutputFlag = 1 if debug else 0
        W = md.addVars(m, lb=a_t.tolist(), ub=b_t.tolist(), name="W")
        z = md.addVars(n, m, lb=0.0, name="z")
        for i in range(n):
            md.addConstr(gp.quicksum(z[i, j] for j in range(m)) == 1.0 / n, name=f"dem_{i}")
        for j in range(m):
            md.addConstr(gp.quicksum(z[i, j] for i in range(n)) == W[j] / n, name=f"sup_{j}")
        md.addConstr(gp.quicksum(float(D_t[i, j]) * z[i, j] for i in range(n) for j in range(m))
                     <= float(epsilon), name="bud")
        md.setObjective(gp.quicksum((float(r_t[j]) - lam) * W[j] for j in range(m)), GRB.MAXIMIZE)
        md.optimize()
        if md.Status != GRB.OPTIMAL:
            raise RuntimeError(f"sn_wass level LP non-optimal (status={md.Status}).")
        return float(md.ObjVal), np.array([W[j].X for j in range(m)])

    lo, hi = float(r_t.min()), float(r_t.max())
    Wbest = b_t.copy()
    if hi - lo < tol:
        return float(hi), Wbest
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val, Wmid = g(mid)
        if val > 0:
            lo = mid; Wbest = Wmid
        else:
            hi = mid
        if hi - lo < tol:
            break
    ratio = float(np.sum(r_t * Wbest) / np.sum(Wbest))
    return ratio, Wbest
