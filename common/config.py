"""Experiment configuration — code 1.1 / common.

A single typed config object that an experiment hands to ``run_experiment``. It carries the three things
that define a comparison run —
  • geometry : how the Wasserstein ground-cost distance D is built — z-score, metric  (common/geometry.py).
               Only the O-W methods use it.
  • support  : how the free-π LP support is built — discretise/snap, mesh, range, rounding (common/support.py).
  • methods  : which methods (and Capped/Uncapped variants) to compare.
plus the experiment-level knobs (arms, capacity, Γ sweep, seeds, sizes, the maximize convention).

Everything round-trips to JSON (``save_json`` / ``load_json``) so each run's exact config is persisted with
its results (SAVING.md provenance) and can be reloaded verbatim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple, Union

# The methods the runner knows how to drive (must match run_experiment.REGISTRY) + their capabilities.
#   The LP methods have Capped/Uncapped variants; KALLUS is PARAMETRIC (softmax θ) ⇒ flat, no capacity
#   (a parametric policy cannot enforce a hard per-arm budget) and deploys via predict_kallus, not Shapley/KNN.
KNOWN_METHODS = ("IPW-O-W", "IPW-O-X", "Hajek-O-W", "Hajek-O-X", "IPW-X-X", "Direct-X-X", "DoublyRobust-X-X", "Kallus",
                 "DoublyRobust-O-W", "DoublyRobust-O-X")   # DR objective over the OW / O uncertainty sets
PARAMETRIC_METHODS = ("Kallus",)                       # flat, no Capped/Uncapped; variant is always ("Flat",)
WASSERSTEIN_METHODS = ("IPW-O-W", "Hajek-O-W", "DoublyRobust-O-W")   # accept the geometry config (zscore / metric)
MAXIMIZE_METHODS = ("Hajek-O-X", "Hajek-O-W", "Kallus") # accept the maximize convention flag
VALID_VARIANTS = ("Capped", "Uncapped")
VALID_METRICS = ("euclidean",)

__all__ = [
    "GeometryConfig", "SupportConfig", "MethodSpec", "ExperimentConfig",
    "KNOWN_METHODS", "WASSERSTEIN_METHODS", "MAXIMIZE_METHODS", "VALID_VARIANTS",
]


@dataclass
class GeometryConfig:
    """Wasserstein ground-cost geometry (common/geometry.py). Used only by the O-W methods."""
    zscore: bool = True            # standardise covariates before distances ⇒ scale-fair ground cost
    metric: str = "euclidean"      # distance metric for D (only euclidean is implemented)

    def __post_init__(self):
        if self.metric not in VALID_METRICS:
            raise ValueError(f"metric {self.metric!r} not supported; valid: {VALID_METRICS}")


@dataclass
class SupportConfig:
    """Free-π LP support construction (common/support.py): the discretisation that de-degenerates the LP."""
    snap: bool = True                                 # discretise (snap to grid) vs raw-X singletons
    mesh: int = 6                                     # grid levels per coordinate when snap=True
    mesh_range: Tuple[float, float] = (-1.0, 1.0)     # (lo, hi) the grid spans, per coordinate
    rounding_digits: int = 6                          # precision at which two support points are one cell

    def __post_init__(self):
        self.mesh_range = (float(self.mesh_range[0]), float(self.mesh_range[1]))
        if int(self.mesh) < 1:
            raise ValueError("mesh must be >= 1")
        self.mesh = int(self.mesh)


@dataclass
class MethodSpec:
    """One method to compare + which capacity variants of it to run.

    Parametric methods (Kallus) are flat: their variant is forced to ``("Flat",)`` regardless of what
    is passed, since a parametric softmax cannot enforce a hard capacity constraint."""
    name: str
    variants: Tuple[str, ...] = VALID_VARIANTS

    def __post_init__(self):
        if self.name not in KNOWN_METHODS:
            raise ValueError(f"unknown method {self.name!r}; known: {KNOWN_METHODS}")
        if self.name in PARAMETRIC_METHODS:
            self.variants = ("Flat",)                  # flat, no capacity — ignore any requested variants
            return
        self.variants = tuple(self.variants)
        bad = [v for v in self.variants if v not in VALID_VARIANTS]
        if bad:
            raise ValueError(f"invalid variant(s) {bad} for {self.name!r}; valid: {VALID_VARIANTS}")


def _coerce_methods(methods: Sequence) -> List[MethodSpec]:
    """Accept names ('IPW-O-W'), MethodSpec objects, or dicts ({'name':.., 'variants':..})."""
    out: List[MethodSpec] = []
    for m in methods:
        if isinstance(m, MethodSpec):
            out.append(m)
        elif isinstance(m, str):
            out.append(MethodSpec(m))
        elif isinstance(m, dict):
            out.append(MethodSpec(m["name"], tuple(m.get("variants", VALID_VARIANTS))))
        else:
            raise TypeError(f"method must be str | MethodSpec | dict, got {type(m).__name__}")
    return out


@dataclass
class ExperimentConfig:
    """The full specification of a comparison run. ``geometry``, ``support`` and ``methods`` are the
    three pieces the runner reads to decide the ground-cost geometry, the discretisation, and what to
    compare; the rest are experiment-level knobs."""
    experiment: str
    n_arms: int
    cap: Tuple[float, ...]
    gamma_true: float
    gammas: Tuple[float, ...]
    seeds: Tuple[int, ...]
    methods: List[MethodSpec]
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    maximize: bool = True                              # reward (True) vs the paper's loss (False) convention
    n_train: int = 300
    n_test: int = 800

    def __post_init__(self):
        if isinstance(self.geometry, dict):
            self.geometry = GeometryConfig(**self.geometry)
        if isinstance(self.support, dict):
            self.support = SupportConfig(**self.support)
        self.methods = _coerce_methods(self.methods)
        self.cap = tuple(float(c) for c in self.cap)
        if len(self.cap) != int(self.n_arms):
            raise ValueError(f"cap length {len(self.cap)} != n_arms {self.n_arms}")
        self.n_arms = int(self.n_arms)
        self.gammas = tuple(float(g) for g in self.gammas)
        self.seeds = tuple(int(s) for s in self.seeds)

    # ---- JSON round-trip (persist the exact config with the results) ----
    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment, "n_arms": self.n_arms, "cap": list(self.cap),
            "gamma_true": self.gamma_true, "gammas": list(self.gammas), "seeds": list(self.seeds),
            "maximize": self.maximize, "n_train": self.n_train, "n_test": self.n_test,
            "geometry": {"zscore": self.geometry.zscore, "metric": self.geometry.metric},
            "support": {"snap": self.support.snap, "mesh": self.support.mesh,
                        "mesh_range": list(self.support.mesh_range),
                        "rounding_digits": self.support.rounding_digits},
            "methods": [{"name": m.name, "variants": list(m.variants)} for m in self.methods],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        d["geometry"] = GeometryConfig(**d.get("geometry", {}))
        d["support"] = SupportConfig(**d.get("support", {}))
        d["methods"] = _coerce_methods(d.get("methods", []))
        return cls(**d)

    def save_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load_json(cls, path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
