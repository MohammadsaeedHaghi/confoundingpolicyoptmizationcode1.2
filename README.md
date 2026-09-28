# code 1.2 — cleaned method library

Cleaned re-organisation of the solver code from `code 1.1` (2026-08-24). Data,
experiment runners, reports and campaign outputs stay in `code 1.1`; this tree holds
only the reusable library code.

## Layout

```
methods/                    one folder per method, .py files directly inside
  IPW-O-W/                  ipw_o_w_uncapped.py, ipw_o_w_capped.py
  DoublyRobust-O-W/         doublyrobust_o_w_{uncapped,capped}.py
  Hajek-O-W/                hajek_o_w_{uncapped,capped}.py
  IPW-O-X/                  ipw_o_x_{uncapped,capped}.py
  DoublyRobust-O-X/         doublyrobust_o_x_{uncapped,capped}.py
  Hajek-O-X/                hajek_o_x_{uncapped,capped}.py
  IPW-X-X/                  ipw_x_x_{uncapped,capped}.py        (non-robust plug-in)
  DoublyRobust-X-X/         doublyrobust_x_x_{uncapped,capped}.py
  Direct-X-X/               direct_x_x_{uncapped,capped}.py     (weights == 1 baseline)
  Oracle/                   oracle_{uncapped,capped}.py
  Kallus/                   kallus.py                            (published baseline)
  SharpHess/                sharp_hess.py                        (published baseline)
common/                     shared infra: propensity/outcome estimation, the MSM box
                            (sensitivity.py), the tight Wasserstein radius
                            (wasserstein_radius.py), geometry, gurobi licence env
extensions/                 policy deployment off the training support
  Shapley/shapley.py        the paper's deployment (used by every campaign)
  KNN/knn.py                k-NN alternative (diagnostic)
```

## Naming

`<Estimator>-<UncertaintySet>`: Estimator in {IPW, DoublyRobust, Hajek, Direct};
uncertainty set `O-W` = odds box INTERSECTED with the Wasserstein ball (the headline
method), `O-X` = odds box only, `X-X` = none (non-robust). `Uncapped` = no per-arm
capacity constraint; `capped` variants add `(1/n) sum_i pi_k(X_i) <= cap_k`.

## Changes vs code 1.1 (layout only — no solver logic touched)

- `Capped/`/`Uncapped/` subfolders flattened into the method folder.
- Each solver's repo-root discovery updated accordingly
  (`Path(__file__).resolve().parents[3]` -> `parents[2]`), so `from common...` imports
  work from this tree without PYTHONPATH tricks.
- `__pycache__`, `.ipynb_checkpoints`, and the per-method HTML notes were not copied.

## Verification (re-run after ANY edit; both must pass)

    sbatch --partition=debug ...   python3 verify_tree.py      # code 1.2 == code 1.1 on every solver (1e-9) + Gamma=1 identity
                                   python3 verify_capped.py    # every capped solver at a non-binding cap (1,1) == its
                                                               # uncapped twin, on 1-D, 2-D and gridded inputs

Status 2026-09-13: both VERIFIED (all 9 solvers 0.0e+00; 27/27 capped==uncapped with dpi = 0).

## Weight conventions (the solvers do NOT estimate anything; weights arrive from `common`)

- `common.ipw_weights_from_data(X, T, K)` -- logistic propensity, clipped to [1e-3, 1-1e-3],
  Hajek-normalised so each arm's weights sum to n. Used by IPW-*, DoublyRobust-* and Hajek-O-W.
  The O-W transport marginals REQUIRE sum_arm w = n, and O-X imposes it as an explicit
  calibration constraint.
- `normalize=False` gives the RAW Horvitz-Thompson weights (all >= 1). Required by Hajek-O-X
  (Charnes-Cooper self-normalised regret; Tan's box is exact on raw weights) and by Kallus.
  Hajek-O-X raises if handed rescaled weights that make the box degenerate.

## Binary treatment only for the Lipschitz policy class

`lipschitz=` constrains pi[1] only, which is correct for n_arms = 2; every solver raises for
n_arms != 2 when it is set. In d > 1 the Lipschitz constraint is imposed on the k = 10
nearest-neighbour pairs (a relaxation of the all-pairs condition; measured 2026-09-13 to hold for
~99.97% of pairs on learned 2-D policies).

## Methodological conventions (settled 2026-09-13)

A. Gamma*. The DGP's confounding level Gamma* is the tight upper bound on the odds ratio between
   the true and the nominal propensity: Gamma* = max_{x,u} max(OR, 1/OR),
   OR = odds(e(x,u)) / odds(e(x)). `common.sensitivity.gamma_star(e_xu, e_x)` computes it for a
   known DGP; it is the smallest Gamma whose box contains every true weight.
B. Order of operations. The nominal propensities are estimated and the weights Hajek-normalised
   first (`common.hajek`); the odds-ratio box 1 + t(w_hat - 1), t in [1/Gamma, Gamma], is then a
   constraint inside the optimisation problem around those nominal weights. Exception:
   Hajek-O-X (and Kallus) self-normalise inside the problem, so they take the raw weights.
C. Hajek-O-X / Hajek-O-W keep their names; they minimise worst-case regret against treat-nobody.
D. code 1.2 is the final library: only what the paper uses. The experimental switches of code 1.1
   (`rationality`/C4, `sharp_cells`, `linear_policy`) were removed on 2026-09-13; they remain in
   code 1.1 with their derivations (`patch_*.py`).

Requires: numpy, scikit-learn, gurobipy (licence via `module load gurobi/12.0.3` on
CARC; `common/gurobi_env.py` handles the token server).
