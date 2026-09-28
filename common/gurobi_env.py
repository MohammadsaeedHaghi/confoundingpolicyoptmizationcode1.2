"""Gurobi licence discovery — code 1.1 / common.

Shared infra for every LP-based method (the O-W, O-X and X-X families, Direct-X-X,
Oracle). Picks the first working licence file so a method can just call this once before
building a model. Ported verbatim from ``code/common/lp_utils.py``.
"""
from __future__ import annotations

import os

import gurobipy as gp

__all__ = ["configure_gurobi_license"]


def configure_gurobi_license() -> None:
    """Point Gurobi at the first licence file that can actually start an environment.

    Tries, in order: ``$GRB_LICENSE_FILE`` → ``~/gurobi.lic`` → ``~/Desktop/gurobi.lic``.
    """
    raw = os.environ.get("GRB_LICENSE_FILE")                 # explicit override, if the user set one
    candidates = []
    if raw:
        candidates.append(os.path.expanduser(raw.strip().strip('"').strip("'")))
    candidates.append(os.path.expanduser("~/gurobi.lic"))    # default grbgetkey location
    candidates.append(os.path.expanduser("~/Desktop/gurobi.lic"))  # common manual location

    seen, paths = set(), []                                  # de-dup by real path (symlink-safe)
    for p in candidates:
        if not p or not os.path.isfile(p):
            continue
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        paths.append(p)
    if not paths:
        raise FileNotFoundError(
            "No Gurobi licence found. Run grbgetkey (~/gurobi.lic), put gurobi.lic on ~/Desktop, "
            "or set GRB_LICENSE_FILE."
        )

    errors = []
    for path in paths:                                       # try each until one actually starts
        os.environ["GRB_LICENSE_FILE"] = path
        env = None
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()                                      # raises if this licence is invalid here
            return
        except gp.GurobiError as exc:
            errors.append(f"{path}: {exc}")
        finally:
            if env is not None:
                env.dispose()
    raise RuntimeError("Gurobi could not use any licence file:\n  " + "\n  ".join(errors))
