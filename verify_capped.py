#!/usr/bin/env python3
"""Verification 2 for the code 1.2 tree: every CAPPED solver with a non-binding cap (1, 1) must
reproduce its UNCAPPED twin -- same worst-case objective (1e-7) and the same policy up to
alternative optima -- on 1-D, 2-D and gridded (discretize=True, mesh=6) inputs, at Gamma = 2,
L = 3, c_eps = 1. Weight conventions follow the campaign runners: Hajek-O-X takes RAW inverse
weights (Charnes-Cooper self-normalised, Tan's box is exact on them); every other weighted solver
takes the per-arm Hajek-normalised weights (the O-W transport requires sum_arm w = n)."""
import sys, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import ipw_weights_from_data, outcome_means


def load(path, fn):
    s = importlib.util.spec_from_file_location("%s_%s" % (fn, path.stem), str(path))
    m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def solvers(name, stem):
    d = ROOT / "methods" / name
    return (load(d / ("%s_uncapped.py" % stem), "solve_%s_uncapped" % stem),
            load(d / ("%s_capped.py" % stem), "solve_%s_capped" % stem))


def case(X):
    rng = np.random.default_rng(3); n = len(X)
    T = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    Y = X[:, 0] + T * X.sum(1) + rng.normal(0, 0.5, n); Y = (Y - Y.mean()) / Y.std()
    w, _ = ipw_weights_from_data(X, T, 2); wraw, _ = ipw_weights_from_data(X, T, 2, normalize=False)
    mu = outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    return T, Y, w, wraw, mu


METHODS = [  # name, stem, how to call (args builder), needs Gamma, O-W (c_eps)
    ("IPW-O-W", "ipw_o_w", lambda X, T, Y, w, wr, mu: (X, T, Y, w), True, True),
    ("DoublyRobust-O-W", "doublyrobust_o_w", lambda X, T, Y, w, wr, mu: (X, T, Y, w, mu), True, True),
    ("Hajek-O-W", "hajek_o_w", lambda X, T, Y, w, wr, mu: (X, T, Y, w), True, True),
    ("IPW-O-X", "ipw_o_x", lambda X, T, Y, w, wr, mu: (X, T, Y, w), True, False),
    ("DoublyRobust-O-X", "doublyrobust_o_x", lambda X, T, Y, w, wr, mu: (X, T, Y, w, mu), True, False),
    ("Hajek-O-X", "hajek_o_x", lambda X, T, Y, w, wr, mu: (X, T, Y, wr), True, False),
    ("IPW-X-X", "ipw_x_x", lambda X, T, Y, w, wr, mu: (X, T, Y, w), False, False),
    ("DoublyRobust-X-X", "doublyrobust_x_x", lambda X, T, Y, w, wr, mu: (X, T, Y, w, mu), False, False),
    ("Direct-X-X", "direct_x_x", lambda X, T, Y, w, wr, mu: (X, T, Y), False, False),
]

rng = np.random.default_rng(0); n = 40
GEOMS = {"1-D raw": (rng.uniform(-1, 1, (n, 1)), False), "2-D raw": (rng.uniform(-1, 1, (n, 2)), False),
         "2-D grid": (rng.uniform(-1, 1, (n, 2)), True)}
bad = 0
print("%-18s %-9s %12s %12s %10s  %s" % ("method", "geometry", "obj uncap", "obj cap(1,1)", "max|dpi|", "verdict"))
for name, stem, args, needs_G, is_ow in METHODS:
    fu, fc = solvers(name, stem)
    for gname, (X, disc) in GEOMS.items():
        T, Y, w, wr, mu = case(X)
        kw = dict(n_arms=2, discretize=disc, mesh=6, lipschitz=3.0)
        if needs_G: kw["Gamma"] = 2.0
        if is_ow: kw.update(zscore=False, c_eps=1.0)
        if name.startswith("Hajek"): kw["maximize"] = True
        try:
            ru = fu(*args(X, T, Y, w, wr, mu), **kw); rc = fc(*args(X, T, Y, w, wr, mu), cap=(1.0, 1.0), **kw)
            ou, oc = getattr(ru, "objective_value", np.nan), getattr(rc, "objective_value", np.nan)
            dpi = float(np.abs(np.asarray(ru.pi) - np.asarray(rc.pi)).max())
            same_obj = abs(ou - oc) <= 1e-7 * max(1.0, abs(ou))
            v = "OK" if (same_obj and dpi <= 1e-6) else ("OK (alt. optimum)" if same_obj else "MISMATCH")
        except Exception as ex:
            ou = oc = dpi = np.nan; v = "ERROR: %s" % str(ex)[:60]
        bad += v.startswith(("MISMATCH", "ERROR"))
        print("%-18s %-9s %12.6f %12.6f %10.2e  %s" % (name, gname, ou, oc, dpi, v))
print("\nCAPPED == UNCAPPED at a non-binding cap:", "VERIFIED" if bad == 0 else "%d FAILURES" % bad)
