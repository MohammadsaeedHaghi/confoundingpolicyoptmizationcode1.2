"""Persistence helpers — code 1.1 / common.

Thin helpers that implement the directory/naming conventions of ``results/SAVING.md`` so every runner
saves results identically:

    results/<experiment>/
        config.json
        data/seed-<NNNN>.npz
        reference/<name>.npz
        <Method>/<Capped|Uncapped>/gtrue-<γ>/{policy.npz, extended.npz, outcomes.csv}
        rosenbaum.csv

Key template inside NPZs: ``<quantity>__G<Γ>__seed<s>`` (Γ rounded to 4 dp, matching the CSV `Gamma`).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Run-level dirs (and config.json) the clean guard must NEVER prune — they belong to the whole run,
# not a single method, and are rewritten each run.
_RUN_LEVEL = ("data", "diagnostics", "reference")

__all__ = ["gkey", "experiment_dir", "method_dir", "save_json", "save_npz", "write_csv",
           "save_raw_draw", "prune_experiment_dir"]


def gkey(quantity: str, Gamma: float, seed: int) -> str:
    """The canonical NPZ key ``<quantity>__G<Γ>__seed<s>`` (Γ at 4 dp, consistent with the CSV)."""
    return f"{quantity}__G{round(float(Gamma), 4)}__seed{int(seed)}"


def experiment_dir(results_root, experiment_slug: str) -> Path:
    """``results/<experiment>/`` (created)."""
    p = Path(results_root) / experiment_slug
    p.mkdir(parents=True, exist_ok=True)
    return p


def method_dir(exp_dir, method: str, cap_label: Optional[str], gamma_true) -> Path:
    """``<exp>/<Method>/<Capped|Uncapped>/gtrue-<γ>/`` (created). ``cap_label=None`` ⇒ flat method
    (e.g. Kallus): ``<exp>/<Method>/gtrue-<γ>/``."""
    g = f"gtrue-{gamma_true:g}"
    p = Path(exp_dir) / method / (cap_label + "/" + g if cap_label else g)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path, obj: dict) -> None:
    """Write a dict as pretty JSON (provenance/config/meta)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=False)


def _jsonable(o):
    """Make numpy / tuples JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_npz(path, **arrays) -> None:
    """``np.savez_compressed`` (arrays keyed by the gkey template)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_csv(path, rows: List[dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    """Write one CSV row per dict (one row per replication cell — never pre-aggregated only)."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_raw_draw(exp_dir, seed: int, *, train: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> None:
    """``data/seed-<NNNN>.npz`` — the raw DGP draws (train_* and test_* fields; §B of SAVING.md)."""
    payload = {f"train_{k}": np.asarray(v) for k, v in train.items()}
    payload.update({f"test_{k}": np.asarray(v) for k, v in test.items()})
    save_npz(Path(exp_dir) / "data" / f"seed-{int(seed):04d}.npz", **payload)


def prune_experiment_dir(exp_dir, planned_leaf_dirs: Sequence, *, dry_run: bool = False) -> List[Path]:
    """Remove STALE method outputs left under ``exp_dir`` by a previous run with a different config.

    ``planned_leaf_dirs`` are the ``<Method>/<Variant>/gtrue-<γ>`` leaves THIS run will write. Any
    ``gtrue-*`` leaf under a method dir that is not in that set is an orphan (a dropped method, a dropped
    Capped/Uncapped variant, or a superseded gamma_true) and is removed; empty method/variant dirs are
    then cleaned up. Run-level dirs (``data``/``diagnostics``/``reference``) and ``config.json`` are NEVER
    touched. With ``dry_run=True`` nothing is deleted — it just returns what WOULD be removed (so a caller
    can warn instead of prune). Returns the list of orphan leaf dirs (removed, or that would be removed)."""
    exp = Path(exp_dir)
    planned = {Path(p).resolve() for p in planned_leaf_dirs}
    orphans: List[Path] = []
    for child in sorted(exp.iterdir()):
        if not child.is_dir() or child.name in _RUN_LEVEL:        # skip files (config.json) + run-level dirs
            continue
        for leaf in sorted(child.rglob("gtrue-*")):               # every result leaf under this method dir
            if leaf.is_dir() and leaf.resolve() not in planned:
                orphans.append(leaf)
                if not dry_run:
                    shutil.rmtree(leaf)
    if not dry_run:                                               # drop now-empty variant/method dirs (bottom-up)
        for child in sorted(exp.iterdir()):
            if not child.is_dir() or child.name in _RUN_LEVEL:
                continue
            for sub in sorted(child.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            if not any(child.iterdir()):
                child.rmdir()
    return orphans
