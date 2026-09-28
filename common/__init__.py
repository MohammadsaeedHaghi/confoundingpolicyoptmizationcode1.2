"""code 1.1 / common — shared pre-algorithm primitives.

Everything a method needs *before* the policy optimiser runs lives here, so all methods
consume identical, calibrated inputs:

  * propensity.py  — nominal propensity estimation  ê(X) = P(T=k | X)   (logistic; estimated, NEVER e_true)
  * propensity_counting.py — empirical-frequency ê(X) by COUNTING within X cells (discrete X)
  * hajek.py       — inverse-propensity weights + Hájek (self-normalising) calibration
  * geometry.py    — covariate distance matrix (Wasserstein cost) + standardisation
  * support.py     — LP support construction: grid-snapping/binning + tie groups
  * sensitivity.py — Marginal Sensitivity Model: Γ* (tight odds-ratio bound of the DGP) + the per-unit MSM box
"""
from __future__ import annotations

from .propensity import (
    fit_propensity_model,
    propensity_matrix,
    factual_propensity,
    propensity_binary,
)
from .propensity_counting import (
    PropensityTable,
    CountingPropensity,
    count_propensity_table,
    count_propensity_matrix,
    count_propensity_binary,
    count_ipw_weights_from_data,
)
from .hajek import (
    hajek_normalize,
    inverse_propensity_weights,
    ipw_weights_from_data,
)
from .geometry import (
    standardize,
    euclidean_distance,
    pairwise_distance_matrix,
    distance_matrix,
)
from .support import (
    grid_levels,
    snap_to_grid,
    tie_groups,
    build_lp_support,
    SupportInfo,
)
from .sensitivity import (
    gamma_star,
    marginal_sensitivity_box,
    augmented_gamma_grid,
)
from .gurobi_env import configure_gurobi_license
from .wasserstein_radius import tight_epsilon_arm, tight_epsilon
from .outcome_model import outcome_means
from .diagnostics import effective_sample_size, propensity_diagnostics
from .config import (
    GeometryConfig,
    SupportConfig,
    MethodSpec,
    ExperimentConfig,
)

__all__ = [
    # propensity
    "fit_propensity_model", "propensity_matrix", "factual_propensity", "propensity_binary",
    # propensity by counting (discrete X)
    "PropensityTable", "CountingPropensity", "count_propensity_table",
    "count_propensity_matrix", "count_propensity_binary", "count_ipw_weights_from_data",
    # hajek
    "hajek_normalize", "inverse_propensity_weights", "ipw_weights_from_data",
    # geometry
    "standardize", "euclidean_distance", "pairwise_distance_matrix", "distance_matrix",
    # support
    "grid_levels", "snap_to_grid", "tie_groups", "build_lp_support", "SupportInfo",
    # sensitivity
    "gamma_star", "marginal_sensitivity_box", "augmented_gamma_grid",
    # solver infra
    "configure_gurobi_license", "tight_epsilon_arm", "tight_epsilon",
    # nuisance
    "outcome_means",
    # diagnostics
    "effective_sample_size", "propensity_diagnostics",
    # experiment config
    "GeometryConfig", "SupportConfig", "MethodSpec", "ExperimentConfig",
]
