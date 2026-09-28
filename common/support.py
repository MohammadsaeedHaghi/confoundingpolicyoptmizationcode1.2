"""Policy support construction (binning) — code 1.1 / common.

The free-π optimisers place one policy variable per *support point* (a location in covariate
space where the policy is defined). On discrete X the support is the distinct X values. On
**continuous X every point is unique** → singletons → the LP degenerates: the box / Wasserstein
robustness has no pooled mass to act on (all methods collapse), and the Wasserstein transport
LP blows up to O(n²).

The fix is to build a coarser support by **snapping** continuous X to a grid, so multiple units
share each cell, and to **tie** the policy across units in the same cell (``tie_same_x``). The
learned policy then becomes a genuine piecewise-constant function of X.

Public API
----------
    grid_levels(n_levels[, lo, hi])              -> (n_levels,) the snap targets per axis
    snap_to_grid(X[, n_levels, lo, hi])          -> X snapped to the grid (same shape, with duplicates)
    tie_groups(X[, rounding_digits])             -> list of index-arrays sharing a (rounded) covariate
    build_lp_support(X[, snap, ...])             -> SupportInfo(support_X, groups, n_cells, levels)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["grid_levels", "snap_to_grid", "tie_groups", "build_lp_support", "SupportInfo"]


def grid_levels(n_levels: int = 6, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """The ``n_levels`` evenly spaced snap targets on ``[lo, hi]`` (per coordinate)."""
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1.")
    return np.linspace(lo, hi, n_levels)


def snap_to_grid(
    X: np.ndarray, n_levels: int = 6, lo: float = -1.0, hi: float = 1.0
) -> np.ndarray:
    """Snap each coordinate of ``X`` to the nearest of ``n_levels`` levels on ``[lo, hi]``.

    Shape is preserved (n rows in, n rows out) — duplicates are intentional; they are what
    ``tie_groups`` / the LP's ``tie_same_x`` then pool. For a 2-D X with ``n_levels=6`` this
    yields ≤ 36 distinct cells.
    """
    X = np.asarray(X, dtype=float)
    levels = grid_levels(n_levels, lo, hi)
    idx = np.abs(X[..., None] - levels).argmin(axis=-1)
    return levels[idx]


def tie_groups(X: np.ndarray, rounding_digits: int = 6) -> List[np.ndarray]:
    """Group row indices of ``X`` that share the same (rounded) covariate value.

    Returns a list of index arrays — one per distinct covariate cell. Units in the same group
    are the ones whose policy is tied (so the policy is a function of X, not of the row index).
    On continuous X with no binning every group has size 1 (the degenerate case).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    keys = {}
    for i, key in enumerate(map(tuple, np.round(X, rounding_digits))):
        keys.setdefault(key, []).append(i)
    return [np.asarray(v, dtype=int) for v in keys.values()]


@dataclass
class SupportInfo:
    """The LP support: where the policy is defined and which units are tied together."""
    support_X: np.ndarray            # (n, d) the support each unit sits on (snapped, with duplicates)
    groups: List[np.ndarray]         # tie groups: index-arrays of units sharing a cell
    n_cells: int                     # number of distinct cells
    levels: Optional[np.ndarray]     # the snap levels used (None if snap=False)


def build_lp_support(
    X: np.ndarray,
    *,
    snap: bool = True,
    n_levels: int = 6,
    lo: float = -1.0,
    hi: float = 1.0,
    rounding_digits: int = 6,
) -> SupportInfo:
    """Build the free-π LP support for ``X``.

    ``snap=True`` (continuous X): snap to an ``n_levels`` grid, then tie units per cell — the
    non-degenerate setup. ``snap=False`` (already-discrete X, or the deliberate "no-grid"
    demonstration): use the raw X as the support (every distinct value its own cell).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    support_X = snap_to_grid(X, n_levels, lo, hi) if snap else X
    groups = tie_groups(support_X, rounding_digits)
    return SupportInfo(
        support_X=support_X,
        groups=groups,
        n_cells=len(groups),
        levels=grid_levels(n_levels, lo, hi) if snap else None,
    )
