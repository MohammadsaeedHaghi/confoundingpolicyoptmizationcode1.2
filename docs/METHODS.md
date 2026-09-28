# What each method in code 1.2 solves

This document states, for every method in `code 1.2/methods`, the optimisation problem the code actually builds and hands to Gurobi (or, for the two published baselines, the procedure it actually runs). It was written by reading the code line by line, not from the paper, so that you can check the implementation against the mathematics. Every program is written in the code's own variables, and every tab lists the file and lines where each piece lives. Where the code does something you might not expect, a **Finding** box says so.

What has already been verified numerically (2026-09-13): code 1.2 reproduces code 1.1 exactly on all nine LP methods (difference 0.0e+00); every capped solver with a non-binding cap equals its uncapped twin (27 of 27 cases); and at Γ = 1, IPW-O-W, IPW-O-X and IPW-X-X give the same optimum (0.314486). Those checks prove the code is *consistent*. Whether it solves the *right problem* is what this document lets you check.

## [Start] Overview

### Notation used on every tab

| Symbol | Meaning | Name in the code |
|---|---|---|
| $n$ | number of training units | `n` |
| $K$ | number of arms; arm 0 is control | `n_arms`, `K` |
| $X_i, T_i, Y_i$ | covariates, observed arm, observed outcome of unit $i$ | `X`, `T`, `Y` |
| $I_k$ | the units observed on arm $k$, $\{i : T_i = k\}$ | `i_by_t[k]` |
| $s_i$ | the support point of unit $i$: $X_i$ itself, or $X_i$ snapped to the grid | `support_X` |
| $\pi_{k,i}$ | decision variable: probability of assigning arm $k$ to unit $i$ | `pi[k, i]` |
| $\hat w_i$ | nominal inverse-propensity weight of unit $i$ (normalised unless stated) | `w_hat`, argument `ips_weights` |
| $W_i$ | the adversary's weight for unit $i$ (inside the uncertainty set) | dualised away; never a variable |
| $[a_i, b_i]$ | the odds-ratio (MSM) box for $W_i$ at sensitivity $\Gamma$ | `a_box`, `b_box` |
| $D_{ij}$ | ground distance $\lVert s_i - s_j \rVert_2$ | `D` |
| $\varepsilon_k$ | Wasserstein radius of arm $k$ | `epsilon[k]` |
| $\hat\mu_k(x)$ | outcome model, estimate of $E[Y \mid X = x, T = k]$ | `muhat`, argument `outcome_means` |
| $r_i$ | residual on the observed arm, $Y_i - \hat\mu_{T_i}(X_i)$ | `resid` |
| $L$ | Lipschitz constant of the policy class | `lipschitz` |

Method names follow `<Estimator>-<UncertaintySet>`. **O-W**: the weights range over the odds box intersected with a Wasserstein covariate-balance ball. **O-X**: the odds box only (plus the per-arm calibration $\sum_{i\in I_k} W_i = n$). **X-X**: no uncertainty set; the nominal weights are used as they are.

### The policy set shared by every LP method

$$
\Pi=\Big\{\pi\ :\ \sum_{k}\pi_{k,i}=1,\ \ 0\le \pi_{k,i}\le 1\ \ \forall i;\quad \pi_{k,i}=\pi_{k,j}\ \text{ whenever } s_i=s_j;\quad \text{[Lipschitz]};\quad \text{[capacity]}\Big\}
$$

The bracketed parts are optional. The Shared inputs tab gives each one exactly as coded.

### The methods at a glance

| Method | Objective | Set the adversary's weights range over | Weights passed in | Solved as | At Γ = 1 |
|---|---|---|---|---|---|
| IPW-O-W | maximise worst-case IPW value | box ∩ per-arm Wasserstein ball | normalised $\hat w$ | one LP (max) | IPW-X-X, when $c_\varepsilon \ge 1$ |
| DoublyRobust-O-W | maximise direct term + worst-case residual term | box ∩ per-arm Wasserstein ball | normalised $\hat w$, $\hat\mu$ | one LP (max) | DoublyRobust-X-X, when $c_\varepsilon \ge 1$ |
| Hajek-O-W | minimise worst-case regret against treat-nobody | box ∩ per-arm Wasserstein ball | normalised $\hat w$ | one LP (min) | same argmax as IPW-X-X |
| IPW-O-X | maximise worst-case IPW value | box ∩ $\{\sum_{I_k} W = n\}$ | normalised $\hat w$ | one LP (max) | IPW-X-X |
| DoublyRobust-O-X | maximise direct term + worst-case residual term | box ∩ $\{\sum_{I_k} W = n\}$ | normalised $\hat w$, $\hat\mu$ | one LP (max) | DoublyRobust-X-X |
| Hajek-O-X | minimise worst-case self-normalised regret | box, self-normalised per arm | **raw** $1/\hat e$ | one LP (Charnes–Cooper, min) | same argmax as IPW-X-X |
| IPW-X-X | maximise Hájek IPW value | none | normalised $\hat w$ | one LP | – |
| DoublyRobust-X-X | maximise AIPW value | none | normalised $\hat w$, $\hat\mu$ | one LP | – |
| Direct-X-X | maximise $\frac1n\sum_i Y_i\pi_{T_i,i}$ | none (weights fixed at 1) | none | one LP | – |
| Kallus | minimise worst-case regret, logistic policy | box, self-normalised per arm | **raw** $1/\hat e$ | subgradient descent (non-convex) | – |
| SharpHess | maximise the efficient sharp lower bound, neural policy | MSM sharp bound | none (fits its own nuisances) | neural nets + Adam | AIPW score |
| Oracle | maximise supplied true values | none | true values | one LP | reference only |

**Why the Γ = 1 column holds.** At Γ = 1 the box collapses to the point $a_i = b_i = \hat w_i$. For O-X the calibration $\sum_{I_k}\hat w = n$ already holds for normalised weights, so the inner problem has the single feasible point $W = \hat w$ and the value is the IPW-X-X value. For O-W the point $W = \hat w$ must also lie in the ball, which holds exactly when $\varepsilon_k \ge \varepsilon_k^{\text{tight}}$, i.e. $c_\varepsilon \ge 1$; with $c_\varepsilon < 1$ the problem is infeasible and the solver raises. For the regret methods the objective becomes $V(\pi_0) - V(\pi)$ with $V(\pi_0)$ a constant, so the minimiser is the IPW-X-X maximiser. For Hajek-O-X on raw weights, the per-arm ratio $\sum_{I_t}\rho_i\tilde w_i/\sum_{I_t}\tilde w_i$ equals $\frac1n\sum_{I_t}\rho_i\hat w_i$, because normalisation only rescales each arm's raw weights to sum to $n$.

### How the experiments call these solvers

- **Outcome.** The runner standardises $Y$ (mean 0, sd 1) before every solve. This matters: with a non-negative, uncentred $Y$ the worst case simply pushes every weight to its lower end, and the box stops discriminating between policies.
- **Weights.** `common.ipw_weights_from_data(X, T, 2)` gives the normalised $\hat w$. Hajek-O-X and Kallus are given `normalize=False`, the raw $\tilde w = 1/\hat e$.
- **Outcome model.** `common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)` (5 folds).
- **Radius.** The runners pass `c_eps` and leave `epsilon=None`, so the solver computes the tight radius itself, on its own support.
- **Deployment.** The learned $\pi_{1,\cdot}$ lives on the training support. Test points are mapped to it by the Shapley extension (after snapping to the grid when `discretize=True`).
- **Γ\*.** The DGP's confounding level is the tight bound on the odds ratio between the true and the nominal propensity, $\Gamma^* = \max_{x,u}\max(\mathrm{OR}, 1/\mathrm{OR})$ with $\mathrm{OR} = \mathrm{odds}(e(x,u))/\mathrm{odds}(e(x))$, computed by `common.sensitivity.gamma_star`.

## [Start] Shared inputs

### Propensity model and weights

Files: `common/propensity.py`, `common/hajek.py`.

The nominal propensity is a logistic regression of $T$ on $X$ (scikit-learn `LogisticRegression`, solver lbfgs, `max_iter=2000`, all other settings at their defaults), clipped to $[10^{-3}, 1-10^{-3}]$ with rows renormalised to sum to 1 (`propensity.py:87`).

$$
\tilde w_i=\frac{1}{\hat e_{T_i}(X_i)}\ \ \ge 1\quad\text{(raw)},\qquad
\hat w_i=\tilde w_i\cdot\frac{n}{\sum_{j\in I_{T_i}}\tilde w_j}\quad\text{(normalised, the default)}
$$

After normalisation every arm's weights sum to $n$: $\sum_{i\in I_k}\hat w_i = n$. This is what the O-X calibration constraint and the O-W transport marginals assume. A normalised weight is below 1 only when $\tilde w_i < \frac1n\sum_{j\in I_{T_i}}\tilde w_j$. Because an arm's average normalised weight is $n/\lvert I_k\rvert \ge 1$, this is rare.

> **Finding.** The logistic fit is L2-regularised: scikit-learn's default is `C = 1`, so this is not the unpenalised maximum-likelihood fit. With $X \in [-1,1]^d$ and $n = 200$ the shrinkage is mild but real: it pulls $\hat e$ toward $1/2$ and narrows the spread of $\hat w$.

### The odds-ratio (MSM) box

File: `common/sensitivity.py:47–67`.

$$
\ell_i = 1+\frac{\hat w_i-1}{\Gamma},\qquad u_i = 1+\Gamma(\hat w_i-1),\qquad a_i=\min(\ell_i,u_i),\qquad b_i=\max(\ell_i,u_i)
$$

**Why this is the right box.** For a raw weight $w = 1/e$ we have $w-1 = (1-e)/e = 1/\mathrm{odds}(e)$. So the true weight $1/e(x,u)$ lies in $\{1 + t(\hat w - 1) : t\in[1/\Gamma,\Gamma]\}$ exactly when $\mathrm{odds}(\hat e)/\mathrm{odds}(e(x,u)) \in [1/\Gamma, \Gamma]$, i.e. when the odds ratio is bounded by $\Gamma$. The smallest $\Gamma$ that covers every unit is $\Gamma^*$.

**Which weights the box is built on.** IPW-O-W, IPW-O-X, DoublyRobust-O-W, DoublyRobust-O-X and Hajek-O-W receive the *normalised* $\hat w$: normalisation happens first, when the nominal propensities are built, and the box is then a constraint inside the optimisation. Hajek-O-X and Kallus receive the *raw* $\tilde w \ge 1$; they normalise inside their own problem.

> **Finding.** When $\hat w_i < 1$ the formula's image is decreasing in $t$, which is why the code takes the min and max. Its lower end $a_i = 1 + \Gamma(\hat w_i - 1)$ becomes **negative** once $\Gamma > 1/(1-\hat w_i)$. In the O-W programs this is harmless: the transport marginal $\sum_i\zeta^k_{ij} = W_j/n$ with $\zeta\ge 0$ already forces $W_j \ge 0$. In the O-X programs the weights are free variables (their dual constraint is an equality), so negative weights would be admitted. Measured on Our DGP and the 2-D lift L4 at $n = 200$ (3 draws each): no unit had $\hat w_i<1$, so no lower end was negative at any Γ up to 150, and clamping the lower end at 0 left IPW-O-X and DoublyRobust-O-X unchanged (identical objective, $\max\lvert\Delta\pi\rvert = 0$). The case is latent in these experiments.

### Support and tying

File: `common/support.py`.

With `discretize=True`, every coordinate of $X_i$ is snapped to the nearest of `mesh` equally spaced levels on `mesh_range` (default 6 levels on $[-1,1]$, so at most 36 cells in 2-D): $s_i = \mathrm{snap}(X_i)$. With `discretize=False`, $s_i = X_i$. Units whose support points agree to 6 decimals form a tie group, and the code adds $\pi_{k,i} = \pi_{k,j}$ for every pair in a group, so the policy is a function of $s$, not of the row index. On continuous raw $X$ every group is a singleton.

### Lipschitz policy class

Present in every LP solver except Oracle. It constrains **arm 1 only**, which is correct for binary treatment; every solver raises when `lipschitz` is set with $K \ne 2$.

$$
\lvert \pi_{1,i}-\pi_{1,j}\rvert \le L\,\lVert s_i-s_j\rVert_2 \qquad \text{for every pair } (i,j) \text{ in the constraint set}
$$

- **1-D (exact).** The pairs are consecutive points after sorting. In one dimension this implies the condition for every pair (chain the inequalities along the sorted order), so nothing is lost.
- **d > 1 (relaxation).** The pairs are $(i,j)$ with $j$ among the `lipschitz_k` nearest neighbours of $i$ (default 10). Pairs outside that graph are only constrained through chains of neighbours, which is weaker than the all-pairs condition.
- Distances are raw Euclidean on the support points (not z-scored), even when the Wasserstein ground cost is z-scored.

> **Caveat.** On a grid, a cell holds many identical points. The $k$ nearest neighbours of a point can then all be copies from its own cell (distance 0), and the constraints only restate the ties. The discretisation runs therefore use `lipschitz_k = 30`.

### Capacity (capped files only)

$$
\frac1n\sum_{i=1}^n\pi_{k,i}\le \mathrm{cap}_k\qquad\forall k
$$

This is the only difference between every `*_capped.py` and its `*_uncapped.py` twin: a diff of every model-building line shows nothing else, and at cap = (1, 1) the two give identical objectives and policies in all 27 tested cases.

### Ground cost and the tight Wasserstein radius

Files: `common/geometry.py`, `common/wasserstein_radius.py:25–52`.

$D_{ij} = \lVert s_i - s_j\rVert_2$ on the support points (z-scoring is available but off by default in the solvers). The tight radius of arm $k$ is the cost of moving the empirical covariate law (mass $1/n$ on every $s_i$) onto the arm-$k$ reweighted law (mass $\hat w_j/n$ on $s_j$, $j\in I_k$):

$$
\varepsilon_k^{\text{tight}}=\min_{\zeta\ge 0}\ \sum_{i=1}^n\sum_{j\in I_k}D_{ij}\,\zeta_{ij}
\quad\text{s.t.}\quad \sum_{j\in I_k}\zeta_{ij}=\frac1n\ \ \forall i,\qquad \sum_{i=1}^n\zeta_{ij}=\frac{\hat w_j}{n}\ \ \forall j\in I_k
$$

The solvers use $\varepsilon_k = c_\varepsilon\cdot\varepsilon_k^{\text{tight}}$. The problem is feasible only because $\sum_{I_k}\hat w = n$. With $c_\varepsilon \ge 1$ the nominal weights $W = \hat w$ lie inside the ball; with $c_\varepsilon < 1$ they are excluded, which is an extra assumption. Read the ball as covariate balance: it keeps the adversary's reweighted arm-$k$ sample within $\varepsilon_k$ of the full sample's covariate distribution.

### Outcome model

File: `common/outcome_model.py`.

For each arm, a regression of $Y$ on $X$ fitted on that arm's units and used to predict every unit: `LinearRegression` for a continuous $Y$, `LogisticRegression` for a binary one. With `cross_fit=True` (what the runners use) the sample is split into 5 random folds and each unit's prediction comes from models fitted on the other folds.

> **Caveat.** A linear $\hat\mu$ is misspecified whenever $E[Y \mid x, t]$ is non-linear. The DR methods then lean on the weighted residual term.

### Deploying a learned policy: the Shapley extension

File: `extensions/Shapley/shapley.py`.

`extract_support` collapses the training policy to its distinct cells (the mean of $\pi$ in each cell, a no-op when the policy is tied). A test point $x$ then gets

$$
\pi^{S}(x)=\min_{k}\ \max_{j}\ A_{jk}(x),\qquad A_{jk}(x)=\frac{d_k\,\hat\pi_j+d_j\,\hat\pi_k}{d_j+d_k},\qquad A_{kk}(x)=\hat\pi_k,\qquad d_j=\lVert x-s_j\rVert_2 ,
$$

with an exact return of $\hat\pi_j$ when $x$ coincides with a support point. Each $A_{jk}$ is a convex combination of two support values, so $\pi^S(x)$ always lies between the smallest and largest learned value.

> **Note.** The docstring calls this "the tightest 1-Lipschitz interpolant". That claim is not checked here. What the code guarantees is exactness at support points and values inside the learned range. The runners in `assets/exp_*` use a vectorised copy (`shapley_md`) of the same formula.

## [O-W] IPW-O-W

| Fact | Value |
|---|---|
| File | `methods/IPW-O-W/ipw_o_w_capped.py` (uncapped: same file minus the capacity constraint) |
| Call | `solve_ipw_o_w_capped(X, T, Y, ips_weights, n_arms, Gamma, cap, discretize, mesh, zscore, c_eps, epsilon, lipschitz, lipschitz_k)` |
| Weights | normalised $\hat w$ |
| Returns | policy $\pi$, worst-case value, all duals $(\beta, \mu, \nu, \gamma, \theta)$ |

### What it optimises

The worst-case IPW value, the worst case taken over the odds box intersected with one Wasserstein ball per arm:

$$
\max_{\pi\in\Pi}\ \ \min_{(W,\zeta)\in\mathcal U(\Gamma,\varepsilon)}\ \ \frac1n\sum_{i=1}^n \pi_{T_i,i}\,Y_i\,W_i
$$

$$
\mathcal U(\Gamma,\varepsilon)=\Big\{(W,\zeta):\ a_i\le W_i\le b_i\ \forall i;\ \ \text{for each arm } k:\ \zeta^k\ge 0,\ \ \sum_{j\in I_k}\zeta^k_{ij}=\tfrac1n\ \forall i,\ \ \sum_{i=1}^n\zeta^k_{ij}=\tfrac{W_j}{n}\ \forall j\in I_k,\ \ \sum_{i}\sum_{j\in I_k}D_{ij}\zeta^k_{ij}\le\varepsilon_k\Big\}
$$

Two consequences of the transport constraints: summing the second marginal over $j\in I_k$ gives $\sum_{j\in I_k}W_j = n$ for every arm (so each arm's term is a Hájek ratio), and $\zeta\ge 0$ gives $W_j\ge 0$.

### The LP the code builds

The inner minimum is an LP in $(W,\zeta)$ for fixed $\pi$. Dualise it with multipliers $\mu_i \ge 0$ for $W_i \ge a_i$, $\nu_i\ge 0$ for $W_i\le b_i$, $\gamma_{k,i}$ free for the first marginal, $\theta_{k,j}$ free for the second, and $\beta_k\ge0$ for the budget. The outer max then merges with the dual max into one LP:

$$
\begin{aligned}
\max_{\pi\in\Pi,\ \beta\ge0,\ \gamma,\ \theta,\ \mu\ge0,\ \nu\ge0}\quad & \sum_{i=1}^n\big(a_i\mu_i-b_i\nu_i\big)+\frac1n\sum_{k}\sum_{i=1}^n\gamma_{k,i}-\sum_k\varepsilon_k\beta_k\\
\text{s.t.}\quad & \tfrac1n\,\pi_{k,i}Y_i+\tfrac1n\,\theta_{k,i}-\mu_i+\nu_i\ \ge\ 0 && \forall k,\ \forall i\in I_k\\
& \beta_k D_{ij}-\gamma_{k,i}-\theta_{k,j}\ \ge\ 0 && \forall k,\ \forall i,\ \forall j\in I_k
\end{aligned}
$$

| Term | Code (capped file) |
|---|---|
| box $[a,b]$ | line 142, `marginal_sensitivity_box(w_hat, Gamma)` |
| radius $\varepsilon_k = c_\varepsilon\varepsilon^{\text{tight}}_k$ | lines 135–136 |
| objective | lines 165–172 |
| Lipschitz, simplex, ties, capacity | lines 175–204 |
| factual-row constraint (dual of $W_i$) | lines 207–211 |
| transport constraint (dual of $\zeta^k_{ij}$) | lines 214–218 |

### How to check the derivation

1. The primal constraint rows and their multipliers: $W_i\ge a_i$ ($\mu_i$), $-W_i\ge -b_i$ ($\nu_i$), $\sum_j\zeta^k_{ij} = 1/n$ ($\gamma_{k,i}$), $\sum_i\zeta^k_{ij} - W_j/n = 0$ ($\theta_{k,j}$), $-\sum D\zeta\ge-\varepsilon_k$ ($\beta_k$).
2. Dual objective: $\sum_i(a_i\mu_i - b_i\nu_i) + \frac1n\sum\gamma - \sum_k\varepsilon_k\beta_k$. This matches lines 165–172.
3. Column of $W_j$ ($j\in I_k$, cost $\frac1n\pi_{k,j}Y_j$): $\mu_j - \nu_j - \theta_{k,j}/n \le \frac1n\pi_{k,j}Y_j$. The inequality treats $W_j$ as sign-constrained, which is valid because transport already forces $W_j\ge0$. Rearranged, this is the factual-row constraint.
4. Column of $\zeta^k_{ij}$ (cost 0): $\gamma_{k,i}+\theta_{k,j}-\beta_kD_{ij}\le 0$. This is the transport constraint.
5. $\pi$ enters only through the right-hand side $\frac1n\pi Y$, linearly, so the whole thing is one LP.
6. Strong duality needs the inner problem feasible: $W=\hat w$ with the tight transport plan is feasible whenever $c_\varepsilon\ge1$ (and the box contains $\hat w$ for every $\Gamma \ge 1$).

### Notes

- If the inner problem is infeasible (e.g. $c_\varepsilon<1$ at $\Gamma = 1$), the dual is unbounded, Gurobi does not return OPTIMAL, and the solver raises `RuntimeError`.
- The objective value is the worst-case value on the training sample, in the units of the $Y$ passed in (standardised in the runners).
- The per-arm balls are separate and every unit belongs to one arm, so the adversary's problem splits into one independent problem per arm.

## [O-W] DoublyRobust-O-W

| Fact | Value |
|---|---|
| File | `methods/DoublyRobust-O-W/doublyrobust_o_w_capped.py` |
| Call | as IPW-O-W, plus `outcome_means` ($n\times K$) |
| Weights | normalised $\hat w$; outcome model $\hat\mu$ |
| Reduces to | IPW-O-W exactly when $\hat\mu\equiv0$ |

### What it optimises

The AIPW value, with only the weighted residual term made robust:

$$
\max_{\pi\in\Pi}\ \ \frac1n\sum_{i=1}^n\sum_{k}\pi_{k,i}\,\hat\mu_k(X_i)\ +\ \min_{(W,\zeta)\in\mathcal U(\Gamma,\varepsilon)}\ \frac1n\sum_{i=1}^n\pi_{T_i,i}\,r_i\,W_i,\qquad r_i=Y_i-\hat\mu_{T_i}(X_i)
$$

### The LP the code builds

The IPW-O-W LP with $Y_i$ replaced by $r_i$ in the factual-row constraint, plus the direct term in the objective:

$$
\begin{aligned}
\max\quad & \frac1n\sum_{i,k}\pi_{k,i}\hat\mu_k(X_i)+\sum_i\big(a_i\mu_i-b_i\nu_i\big)+\frac1n\sum_{k,i}\gamma_{k,i}-\sum_k\varepsilon_k\beta_k\\
\text{s.t.}\quad & \tfrac1n\,\pi_{k,i}\,r_i+\tfrac1n\,\theta_{k,i}-\mu_i+\nu_i\ \ge\ 0 \quad\forall k,\ i\in I_k;\qquad \beta_kD_{ij}-\gamma_{k,i}-\theta_{k,j}\ge 0
\end{aligned}
$$

| Term | Code (capped file) |
|---|---|
| residual $r_i$ | line 114 |
| objective (direct + dual) | lines 158–167 |
| factual-row constraint with $r_i$ | lines 202–205 |
| transport constraint | lines 208–212 |

### Why robustifying only the residual is enough

If the true weights $W^*_i = 1/e_{T_i}(X_i, U_i)$ were known, then for **any** $\hat\mu$, $E\big[\sum_k\pi_k\hat\mu_k(X) + \pi_T W^*(Y-\hat\mu_T(X))\big] = V(\pi)$, because $W^*$ reweights each arm to its potential-outcome distribution. So whenever the true weights lie in $\mathcal U$, the robust objective is a lower bound on the value, whatever the outcome model. The outcome model does not have to be unconfounded: it only changes how tight the bound is. The box width multiplies $\lvert r_i\rvert$ instead of $\lvert Y_i\rvert$, so a good $\hat\mu$ makes the worst case less pessimistic.

> **Caveat.** That argument uses the unnormalised true weights. With normalised weights and a box built on them (the project's convention), it holds approximately, not exactly.

## [O-W] Hajek-O-W

| Fact | Value |
|---|---|
| File | `methods/Hajek-O-W/hajek_o_w_capped.py` |
| Call | as IPW-O-W, plus `maximize` (default `True`; `False` flips $Y\to -Y$) |
| Weights | normalised $\hat w$ (unlike Hajek-O-X, which takes raw weights) |
| Returns | policy, worst-case regret ($\le 0$), duals |

### What it optimises

The worst-case **regret** against the treat-nobody policy $\pi^0$ ($\pi^0_{0,i} = 1$ for everyone), over the same set as IPW-O-W:

$$
\min_{\pi\in\Pi}\ \ \max_{(W,\zeta)\in\mathcal U(\Gamma,\varepsilon)}\ \ \frac1n\sum_{i=1}^n\big(\mathbf 1[T_i=0]-\pi_{T_i,i}\big)\,Y_i\,W_i \;=\; \min_{\pi}\max_{W}\ \big[V(\pi^0,W)-V(\pi,W)\big]
$$

with $V(\pi,W)=\frac1n\sum_i\pi_{T_i,i}Y_iW_i$. Because the transport marginals force $\sum_{I_k}W = n$, each arm's contribution equals a self-normalised (Hájek) ratio $\sum_{I_k}\rho_iW_i/\sum_{I_k}W_i$. That is where the name comes from: it is not a Hájek *value* estimator.

### The LP the code builds

Dualise the inner **max** with $p_i\ge0$ for $W_i\le b_i$, $q_i\ge0$ for $W_i\ge a_i$, $\gamma,\theta$ free and $\beta_k\ge0$:

$$
\begin{aligned}
\min_{\pi\in\Pi,\ \beta\ge0,\ \gamma,\ \theta,\ p\ge0,\ q\ge0}\quad & \sum_{i}\big(b_ip_i-a_iq_i\big)+\frac1n\sum_{k,i}\gamma_{k,i}+\sum_k\varepsilon_k\beta_k\\
\text{s.t.}\quad & p_i-q_i-\tfrac1n\,\theta_{k,i}+\tfrac1n\,\pi_{k,i}Y_i=\tfrac1n\,\mathbf 1[k=0]\,Y_i && \forall k,\ \forall i\in I_k\\
& \beta_kD_{ij}+\gamma_{k,i}+\theta_{k,j}\ \ge\ 0 && \forall k,\ \forall i,\ \forall j\in I_k
\end{aligned}
$$

| Term | Code (capped file) |
|---|---|
| objective | lines 174–180 |
| factual-row equality (dual of $W_i$) | lines 217–221 |
| transport constraint | lines 224–228 |

**Check.** The column of $W_j$ is $p_j - q_j - \theta_{k,j}/n = \frac1n(\mathbf 1[k=0]-\pi_{k,j})Y_j$, an equality because $W$ is declared free here. That is equivalent to the IPW-O-W form, since transport forces $W\ge 0$ anyway. The column of $\zeta$: $\gamma_{k,i}+\theta_{k,j}+\beta_kD_{ij}\ge0$ (a max problem's dual, so the sign of $\beta$'s term flips relative to IPW-O-W).

### Notes

- $\pi = \pi^0$ is feasible and gives regret 0, so the optimum is $\le 0$: the method never does worse than treating nobody *in the worst case*. At large Γ the optimum is often $\pi^0$ itself, which explains why its value so often equals treat-nobody.
- Its box is built on normalised weights, while Hajek-O-X's is built on raw weights. The two sets are close but not identical at the same Γ, because the box $1 + t(w - 1)$ is not invariant to rescaling $w$.

## [O-X] IPW-O-X

| Fact | Value |
|---|---|
| File | `methods/IPW-O-X/ipw_o_x_capped.py` |
| Call | as IPW-O-W without `c_eps`, `epsilon`, `zscore`, `metric` |
| Weights | normalised $\hat w$ |
| Returns | policy, worst-case value, duals $(\mu, \nu, \beta)$ — here $\beta$ is the calibration dual, free in sign |

### What it optimises

$$
\max_{\pi\in\Pi}\ \ \min_{W}\ \ \frac1n\sum_{i=1}^n\pi_{T_i,i}\,Y_i\,W_i\qquad\text{s.t.}\quad a_i\le W_i\le b_i\ \ \forall i,\qquad \sum_{i\in I_k}W_i=n\ \ \forall k
$$

For fixed $\pi$ the inner problem is a fractional knapsack per arm: starting from $W = a$, the adversary spends the remaining budget $n - \sum_{I_k}a_i$ raising weights toward $b_i$ on the units with the smallest $\pi_{k,i}Y_i$ first.

### The LP the code builds

Multipliers $\mu_i\ge0$ ($W_i\ge a_i$), $\nu_i\ge0$ ($W_i\le b_i$), $\beta_k$ free (calibration); $W$ is a free variable, so its dual constraint is an equality:

$$
\begin{aligned}
\max_{\pi\in\Pi,\ \mu\ge0,\ \nu\ge0,\ \beta}\quad & \sum_{i=1}^n\big(a_i\mu_i-b_i\nu_i\big)+n\sum_k\beta_k\\
\text{s.t.}\quad & \mu_i-\nu_i+\beta_{T_i}=\tfrac1n\,\pi_{T_i,i}\,Y_i\qquad\forall i
\end{aligned}
$$

| Term | Code (capped file) |
|---|---|
| box | line 101 |
| objective | lines 116–121 |
| stationarity (dual of $W_i$) | lines 125–127 |
| policy constraints | lines 130–159 |

> **Finding.** Because $W$ is free, a negative lower end $a_i<0$ (possible for $\hat w_i<1$ at large Γ; see Shared inputs) would let the adversary put a negative weight on unit $i$; the O-W programs cannot. Measured: it never happened in our designs, and clamping at 0 changes nothing (Findings tab, item 1).

## [O-X] DoublyRobust-O-X

| Fact | Value |
|---|---|
| File | `methods/DoublyRobust-O-X/doublyrobust_o_x_capped.py` |
| Weights | normalised $\hat w$; outcome model $\hat\mu$ |
| Reduces to | IPW-O-X exactly when $\hat\mu\equiv0$ |

$$
\max_{\pi\in\Pi}\ \frac1n\sum_{i,k}\pi_{k,i}\hat\mu_k(X_i)+\min_{W}\Big\{\frac1n\sum_i\pi_{T_i,i}\,r_i\,W_i\ :\ a\le W\le b,\ \sum_{I_k}W=n\Big\}
$$

The LP is IPW-O-X's with $Y_i$ replaced by $r_i$ and the direct term added:

$$
\max\ \ \frac1n\sum_{i,k}\pi_{k,i}\hat\mu_k(X_i)+\sum_i\big(a_i\mu_i-b_i\nu_i\big)+n\sum_k\beta_k\quad\text{s.t.}\quad \mu_i-\nu_i+\beta_{T_i}=\tfrac1n\,\pi_{T_i,i}\,r_i\ \ \forall i
$$

| Term | Code (capped file) |
|---|---|
| residual | line 101 |
| objective | lines 124–132 |
| stationarity | lines 136–138 |

The argument for robustifying only the residual is the same as on the DoublyRobust-O-W tab, and so is the negative-weight finding on the IPW-O-X tab.

## [O-X] Hajek-O-X

| Fact | Value |
|---|---|
| File | `methods/Hajek-O-X/hajek_o_x_capped.py` |
| Weights | **raw** $\tilde w_i = 1/\hat e_{T_i}(X_i)\ge1$; the solver raises if any $a_i \le 0$ (lines 110–111) |
| Returns | policy, worst-case regret ($\le 0$), per-arm levels $\lambda_t$, duals $u, v$ |

### What it optimises

Kallus & Zhou's self-normalised worst-case regret against treat-nobody, one ratio per arm:

$$
\min_{\pi\in\Pi}\ \sum_{t}\hat Q_t(\pi),\qquad
\hat Q_t(\pi)=\max_{a\le W\le b}\ \frac{\sum_{i\in I_t}\rho_i\,W_i}{\sum_{i\in I_t}W_i},\qquad
\rho_i=\big(\mathbf 1[t=0]-\pi_{t,i}\big)\,Y_i
$$

There is no calibration constraint: the denominator floats with $W$.

### The LP the code builds

Charnes–Cooper turns each ratio into an LP: $\hat Q_t = \max_{w\ge0,\,s\ge0}\sum_{I_t}\rho_iw_i$ s.t. $\sum_{I_t}w_i = 1$, $s\,a_i\le w_i\le s\,b_i$. Its dual is $\hat Q_t = \min\lambda_t$ s.t. $\lambda_t - u_i + v_i\ge\rho_i$, $\sum_{I_t}(a_iu_i - b_iv_i)\ge0$, $u,v\ge0$. Merging with the outer min:

$$
\begin{aligned}
\min_{\pi\in\Pi,\ \lambda,\ u\ge0,\ v\ge0}\quad & \sum_t\lambda_t\\
\text{s.t.}\quad & v_i-u_i+\lambda_{T_i}+Y_i\,\pi_{T_i,i}\ \ge\ \mathbf 1[T_i=0]\,Y_i && \forall i\\
& \sum_{i\in I_t}\big(a_iu_i-b_iv_i\big)\ \ge\ 0 && \forall t
\end{aligned}
$$

| Term | Code (capped file) |
|---|---|
| objective $\sum_t\lambda_t$ | line 131 |
| per-unit constraint (dual of $w_i$) | lines 135–138 |
| per-arm constraint (dual of $s$) | lines 140–142 |

**Check.** In the Charnes–Cooper primal the rows are $\sum w = 1$ (multiplier $\lambda$, free), $-w_i + s\,a_i\le 0$ ($u_i$), $w_i - s\,b_i\le0$ ($v_i$). Column of $w_i$: $\lambda - u_i + v_i\ge\rho_i$. Column of $s$ (cost 0): $\sum(a_iu_i - b_iv_i)\ge0$. Charnes–Cooper needs a positive denominator, hence $a_i>0$, hence raw weights.

### Notes

- Regret $\le 0$ and the treat-nobody floor, as for Hajek-O-W.
- At Γ = 1 it has the same argmax as IPW-X-X (see the Overview).

## [X-X] IPW-X-X

| Fact | Value |
|---|---|
| File | `methods/IPW-X-X/ipw_x_x_capped.py` |
| Weights | normalised $\hat w$ |
| Returns | policy and value; no duals |

$$
\max_{\pi\in\Pi}\ \ \frac1n\sum_{i=1}^n\hat w_i\,Y_i\,\pi_{T_i,i}\;=\;\sum_k\frac{\sum_{i\in I_k}\hat w_iY_i\pi_{k,i}}{\sum_{i\in I_k}\hat w_i}
$$

The equality holds because $\sum_{I_k}\hat w = n$, so this is the Hájek (self-normalised) IPW value. Objective: lines 104–108. It is the non-robust point that IPW-O-W and IPW-O-X reduce to at Γ = 1.

## [X-X] DoublyRobust-X-X

| Fact | Value |
|---|---|
| File | `methods/DoublyRobust-X-X/doublyrobust_x_x_capped.py` |
| Weights | normalised $\hat w$; outcome model $\hat\mu$ |
| Note | accepts a `Gamma` argument for signature parity and **ignores it** |

$$
\max_{\pi\in\Pi}\ \ \frac1n\sum_{i=1}^n\Big[\sum_k\pi_{k,i}\,\hat\mu_k(X_i)+\hat w_i\,r_i\,\pi_{T_i,i}\Big]
$$

The standard AIPW value at the nominal weights (objective: lines 109–115). DoublyRobust-O-W and DoublyRobust-O-X reduce to it at Γ = 1.

## [X-X] Direct-X-X

| Fact | Value |
|---|---|
| File | `methods/Direct-X-X/direct_x_x_capped.py` |
| Weights | none (every weight fixed at 1) |

$$
\max_{\pi\in\Pi}\ \ \frac1n\sum_{i=1}^n Y_i\,\pi_{T_i,i}
$$

Objective: lines 100–102.

> **Finding.** This is IPW with every weight set to 1, and it does **not** estimate the policy value, even without confounding. Its population target is $E[\pi_T(X)\,Y] = \sum_k E[e_k(X)\,\pi_k(X)\,E(Y \mid X, T=k)]$, i.e. each arm's outcome weighted by how often the logging policy chose that arm. In a cell it prefers the arm whose observed outcomes have the larger *sum*, which mixes outcome size with how many units took that arm. It is not the "naive" regression rule: the runners' separate `naive` baseline is $\pi_1(x) = \mathbf 1[\hat\mu_1(x) > \hat\mu_0(x)]$.

## [Baselines] Kallus

| Fact | Value |
|---|---|
| File | `methods/Kallus/kallus.py`, `fit_kallus_paper` (lines 363–438), inner solver lines 312–341 |
| Source | a port of Kallus & Zhou's repository (confounding-robust-policy-improvement), their synthetic driver's settings |
| Weights | **raw** $\tilde w\ge 1$ (raises otherwise) |
| Policy | logistic, $\pi_1(x)=\sigma(\theta^\top[x,1])$; binary treatment only |

### What it optimises

The same objective as Hajek-O-X, restricted to logistic policies. With $y^{\text{loss}}_i = -Y_i$ (because `maximize=True`) and $\mathrm{sgn}_i = +1$ for treated, $-1$ for control:

$$
\min_{\theta\in\mathbb R^{d+1}}\ \sum_{s\in\{\text{treated},\,\text{control}\}}\ \max_{a\le W\le b}\ \frac{\sum_{i\in I_s}c_i(\theta)\,W_i}{\sum_{i\in I_s}W_i},\qquad c_i(\theta)=\mathrm{sgn}_i\,y^{\text{loss}}_i\,\sigma\big(\theta^\top[X_i,1]\big)
$$

with the box on raw weights, $a_i = 1 + (\tilde w_i - 1)/\Gamma$, $b_i = 1 + (\tilde w_i - 1)\Gamma$. For a treated unit $c_i = -Y_i\pi_1$ and for a control unit $c_i = Y_i\pi_1$, which is exactly Hajek-O-X's $\rho_i$ for two arms.

### How it is solved (not an LP)

1. **Inner max.** Sort by $c_i$; the maximiser puts weight $a$ below a threshold and $b$ above it. The threshold is found by ternary search on the ratio, as in their `find_opt_weights_shorter`.
2. **Outer step.** Subgradient $g = \sum_i y^{\text{loss}}_i\,\mathrm{sgn}_i\,\bar W_i\,\pi_{1,i}(1-\pi_{1,i})\,[X_i,1]$, where $\bar W$ is each arm's worst-case weights normalised to sum to 1 and then renormalised globally (each arm therefore carries total weight $1/2$, which only rescales the step). By Danskin's theorem this is a valid subgradient of the max.
3. **Step size.** Armijo backtracking (start 1, factor 0.2, $\beta = 10^{-4}$, up to 20 tries), falling back to $1/\sqrt{k+1}$.
4. **Rounds.** $N = \mathrm{clip}(\lfloor G^2D^2/0.05^2\rfloor, 50, 200)$, with their $D$ and $G$ formulas (their $G$ uses the raw dimension $d$, reproduced as-is).
5. **Restarts.** 15; restart 0 starts at $\theta = (0,\dots,0,-1000)$, i.e. at treat-nobody; the others at $0.25\cdot N(0, I)$. Each restart is scored by the mean loss over its rounds and returns the average of its iterates (Polyak averaging). The best restart wins.

> **Caveat.** The problem is non-convex in $\theta$, so this is a heuristic with restarts, not a certified optimum. The single deliberate deviation from their code is that numpy is seeded per restart (theirs is effectively unseeded). At large Γ the treat-nobody start usually wins: that is the regret floor, not an optimiser failure (checked 2026-09-13).

## [Baselines] SharpHess

| Fact | Value |
|---|---|
| File | `methods/SharpHess/sharp_hess.py`: `hess_paper` (lines 528–555), `nuisances_nn_paper` (496–525), `scores` (101–128), `_MLP` (312–) |
| Source | Hess, Frauen, Melnychuk & Feuerriegel (ICLR 2026), replicating their repository's code |
| Policy | a {64, 32} ReLU network with sigmoid output |

### What it computes

**Step 1, sign and scale.** Work with $Y^p = -Y$ (their convention is "lower is better"), standardised to mean 0, sd 1.

**Step 2, split.** A random 50/50 split into halves $A$ (nuisances) and $B$ (policy).

**Step 3, nuisances on $A$.** Each is a {64, 32} MLP trained with Adam (lr $10^{-3}$, batch 64, at most 300 epochs, early stopping with patience 10 on a 20% validation split):

- $\hat e(x)$, the propensity (binary cross-entropy; numerical floor $10^{-6}$);
- $\hat q(x,a)$, the $\alpha^+$-quantile of $Y^p$ given $(x,a)$, with $\alpha^+ = \Gamma/(1+\Gamma)$ (pinball loss on $(x,\text{one-hot }a)$);
- $\hat\mu^+(x,a) = E[Y^p\,\mathbf 1\{Y^p\le \hat q\}]$ and $\hat{\bar\mu}^+(x,a) = E[Y^p\,\mathbf 1\{Y^p\ge \hat q\}]$ (MSE on masked targets, with the quantile net frozen).

**Step 4, scores on $B$** (their Eq. 15), with $b^+ = 1-\Gamma$, $b^- = 1-1/\Gamma$, $c^\pm(a,x) = b^\pm e(a,x) + \Gamma^{\pm1}$ and $Q = c^-\mu^+ + c^+\bar\mu^+$:

$$
g_i(a)=Q_{a,i}-e_{a,i}\big(b^-\mu^+_{a,i}+b^+\bar\mu^+_{a,i}\big)+\mathbf 1[T_i=a]\Big\{b^-\mu^+_{a,i}+b^+\bar\mu^+_{a,i}+\frac{(c^-_{a,i}-c^+_{a,i})\,q_{a,i}(\Delta_i-\alpha^+)+c^-_{a,i}(Y^p_i\Delta_i-\mu^+_{a,i})+c^+_{a,i}(Y^p_i\bar\Delta_i-\bar\mu^+_{a,i})}{e_{a,i}}\Big\}
$$

with $\Delta_i = \mathbf 1\{Y^p_i\le q_{T_i,i}\}$ and $\bar\Delta_i = \mathbf 1\{Y^p_i\ge q_{T_i,i}\}$. The scores are then negated, turning an upper bound on the loss into a lower bound on the value on our scale.

**Step 5, policy on $B$.** A network $f$ trained to minimise $-\frac1{\lvert B\rvert}\sum_{i\in B}\sigma(f(X_i))\,\big(g_i(1)-g_i(0)\big)$, i.e. to maximise the estimated sharp lower bound $\frac1{\lvert B\rvert}\sum_i[\pi_1 g_i(1) + (1-\pi_1)g_i(0)]$ ($g_i(0)$ does not depend on the policy). Deployment: $\pi_1(x) = \sigma(f(x))$.

### Notes

- At Γ = 1: $b^\pm = 0$, $c^\pm = 1$, $\alpha^+ = 1/2$, and the score reduces to the AIPW score. The file contains numerical tests of this identity.
- The policy is learned from half the sample (100 units at $n = 200$), has no Lipschitz constraint, and is noticeably seed-sensitive. Use at least 10 seeds.

## [Reference] Oracle

| Fact | Value |
|---|---|
| File | `methods/Oracle/oracle_capped.py`, `oracle_uncapped.py` |
| Input | a value matrix `values` ($n\times K$), e.g. true potential outcomes or true means |

$$
\max_{\pi\in\Pi_{\text{no Lipschitz}}}\ \ \frac1n\sum_{i=1}^n\sum_k\pi_{k,i}\,V_{i,k}
$$

A reference ceiling, not a method. Without capacity it reduces to assigning each cell to the arm with the largest cell-average value. No current runner calls it: the experiment runners compute the oracle directly from the DGP, $\pi^*(x) = \mathbf 1[E(\tau\mid x) > 0]$.

## [Findings] Findings

Every item below was found by reading the code for this document. They are listed with how much they matter.

### Checked and correct

- **The six robust LPs.** For IPW-O-W, DoublyRobust-O-W, Hajek-O-W, IPW-O-X, DoublyRobust-O-X and Hajek-O-X, the dual was re-derived term by term from the stated inner problem, and every objective term, constraint and sign in the code matches (the "How to check" steps on each tab).
- **The three non-robust LPs and the Oracle** are the stated linear objectives over $\Pi$.
- **Capped vs uncapped.** They differ only in the capacity constraint (a diff of every model line, and 27 of 27 numerical cases).
- **Γ = 1 reductions.** Each is exact for the reasons given on the Overview tab; IPW-O-W = IPW-O-X = IPW-X-X was confirmed numerically.
- **The box.** It is Tan's odds-ratio interval for a raw weight, and $\Gamma^*$ as computed is the smallest Γ that contains every true weight.

### Worth your attention

1. **O-X could admit negative weights (latent).** For a unit with normalised $\hat w_i<1$ the box's lower end is negative once $\Gamma>1/(1-\hat w_i)$, and in IPW-O-X / DoublyRobust-O-X the weight variable is free. O-W is not affected (transport forces $W\ge0$), nor are Hajek-O-X and Kallus (raw weights $\ge 1$). *Measured (debug job 11971463):* on Our DGP and L4 at $n = 200$, 3 draws each, no unit had $\hat w_i<1$; at Γ ∈ {2.9, 5, 30, 150}, clamping the lower end at 0 gave identical objectives and policies for both O-X methods. A one-line guard, `a = max(a, 0)` in the O-X solvers, would make it impossible; it is not applied, to keep code 1.2 identical to code 1.1.
2. **Two weight conventions.** The box is built on normalised weights for five methods and on raw weights for Hajek-O-X and Kallus. At the same Γ these are close but different sets, because $1 + t(w-1)$ is not invariant to rescaling $w$. This is the agreed convention (normalise first, then box); state it in the paper.
3. **The propensity model is L2-regularised** (scikit-learn default `C = 1`), not the unpenalised MLE.
4. **Direct-X-X is not a value estimator.** It maximises a propensity-weighted sum of factual outcomes (see its tab). If the paper calls it "naive", say what it computes.

### By design, but say it in the paper

5. **Regret methods have a treat-nobody floor.** Hajek-O-W, Hajek-O-X and Kallus minimise worst-case regret against treat-nobody, so their optimum is never worse than treating nobody in the worst case, and at large Γ it often *is* treating nobody.
6. **The Lipschitz class in $d>1$ is a $k$-nearest-neighbour relaxation**, and on a grid the neighbours can be copies of the same cell. The 1-D class is exact.
7. **Kallus is a non-convex heuristic** (15 restarts, subgradient descent, Polyak averaging); **SharpHess learns from half the sample** with neural networks. Both are implemented as their authors' code does it.

### Harmless

8. **DoublyRobust-X-X ignores its `Gamma` argument.** It is kept only so the call signature matches.
9. **The Shapley extension's "tightest 1-Lipschitz" docstring claim is unverified;** exactness at support points holds.
10. **The mathematical notes** (`<Method>.html`) that the docstrings point to live in code 1.1, not code 1.2.
