#!/usr/bin/env python3
"""Verification for the code 1.2 tree (run once after any re-organisation):
1. every solver file imports and exposes its solve_* function;
2. on a tiny synthetic problem, each code 1.2 solver's objective and policy match the
   code 1.1 original to 1e-9 (same inputs, Gamma = 2, L = 3, c_eps = 1);
3. the Gamma = 1 identity holds: IPW-O-W == IPW-O-X == IPW-X-X."""
import sys, importlib.util
from pathlib import Path
import numpy as np

NEW = Path(__file__).resolve().parent
OLD = Path("/home1/haghim/code 1.1")


def load(path, fn):
    s = importlib.util.spec_from_file_location(fn + "_" + path.parent.name, str(path))
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    return getattr(m, fn)


rng = np.random.default_rng(0)
n = 60
X = rng.uniform(-1, 1, n).reshape(-1, 1)
T = (rng.uniform(size=n) < 0.5).astype(int)
Y = rng.normal(size=n)
sys.path.insert(0, str(NEW))
from common import ipw_weights_from_data, outcome_means, pairwise_distance_matrix
from common.wasserstein_radius import tight_epsilon
w, _ = ipw_weights_from_data(X, T, 2)
wraw, _ = ipw_weights_from_data(X, T, 2, normalize=False)
mu = outcome_means(X, T, Y, n_arms=2, cross_fit=True)
D = pairwise_distance_matrix(X)
eps = tuple(tight_epsilon(D, T, w, 2, is_distance=True, c_eps=1.0))

CASES = [
    ("IPW-O-X", "ipw_o_x_uncapped", "solve_ipw_o_x_uncapped",
     lambda f, G: f(X, T, Y, w, n_arms=2, Gamma=G, discretize=False, lipschitz=3.0)),
    ("DoublyRobust-O-X", "doublyrobust_o_x_uncapped", "solve_doublyrobust_o_x_uncapped",
     lambda f, G: f(X, T, Y, w, mu, n_arms=2, Gamma=G, discretize=False, lipschitz=3.0)),
    ("Hajek-O-X", "hajek_o_x_uncapped", "solve_hajek_o_x_uncapped",
     lambda f, G: f(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, discretize=False,
                    lipschitz=3.0)),
    ("IPW-O-W", "ipw_o_w_uncapped", "solve_ipw_o_w_uncapped",
     lambda f, G: f(X, T, Y, w, n_arms=2, Gamma=G, discretize=False, lipschitz=3.0,
                    zscore=False, epsilon=eps)),
    ("DoublyRobust-O-W", "doublyrobust_o_w_uncapped", "solve_doublyrobust_o_w_uncapped",
     lambda f, G: f(X, T, Y, w, mu, n_arms=2, Gamma=G, discretize=False, lipschitz=3.0,
                    zscore=False, epsilon=eps)),
    ("Hajek-O-W", "hajek_o_w_uncapped", "solve_hajek_o_w_uncapped",
     lambda f, G: f(X, T, Y, w, n_arms=2, Gamma=G, maximize=True, discretize=False,
                    lipschitz=3.0, zscore=False, epsilon=eps)),
    ("IPW-X-X", "ipw_x_x_uncapped", "solve_ipw_x_x_uncapped",
     lambda f, G: f(X, T, Y, w, n_arms=2, discretize=False, lipschitz=3.0)),
    ("DoublyRobust-X-X", "doublyrobust_x_x_uncapped", "solve_doublyrobust_x_x_uncapped",
     lambda f, G: f(X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=3.0)),
    ("Direct-X-X", "direct_x_x_uncapped", "solve_direct_x_x_uncapped",
     lambda f, G: f(X, T, Y, n_arms=2, discretize=False, lipschitz=3.0)),
]

vals1 = {}
for meth, stem, fn, call in CASES:
    fnew = load(NEW / "methods" / meth / (stem + ".py"), fn)
    fold = load(OLD / "methods" / meth / "Uncapped" / (stem + ".py"), fn)
    rn, ro = call(fnew, 2.0), call(fold, 2.0)
    dv = abs(rn.objective_value - ro.objective_value)
    dp = float(np.max(np.abs(np.asarray(rn.pi) - np.asarray(ro.pi))))
    assert dv < 1e-9 and dp < 1e-7, (meth, dv, dp)
    print("%-18s new==old  (dobj %.1e, dpi %.1e)" % (meth, dv, dp), flush=True)
    vals1[meth] = call(fnew, 1.0).objective_value

# capped variants import check
for meth in ("IPW-O-W", "IPW-O-X", "DoublyRobust-O-W", "DoublyRobust-O-X", "Hajek-O-W",
             "Hajek-O-X", "IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "Oracle"):
    for f in (NEW / "methods" / meth).glob("*_capped.py"):
        fn = "solve_" + f.stem
        load(f, fn)
print("all capped variants import OK", flush=True)

g1 = [vals1["IPW-O-W"], vals1["IPW-O-X"], vals1["IPW-X-X"]]
assert max(g1) - min(g1) < 1e-8, g1
print("Gamma=1 identity holds: IPW-O-W == IPW-O-X == IPW-X-X == %.6f" % g1[0])
print("CODE 1.2 TREE VERIFIED")
