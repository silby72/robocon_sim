"""P4 sweep-engine acceptance + unit tests (依頼 §5).

Numbered ``acceptance N:`` comments map to the 5 P4 criteria. run_one functions
are module-level so multiprocessing can pickle them.
"""
from __future__ import annotations

import numpy as np

from omni_sim_core.sweep import (SweepSpec, run_sweep, find_boundary,
                                 bisect_boundary)


# -- module-level run_one functions (picklable) ----------------------------- #
def run_rng(params, rng):
    """Metric depends on the per-condition RNG -> proves RNG determinism."""
    return {"noise": float(rng.standard_normal()), "x": params["x"]}


def run_boundary(params, rng):
    """Synthetic: min_clearance crosses 0.15 at width == 0.65."""
    return {"min_clearance": 0.8 - params["footprint_width"]}


def run_flaky(params, rng):
    if params["x"] == 3:
        raise ValueError("boom")
    return {"y": params["x"] * 2}


# --------------------------------------------------------------------------- #
def test_acceptance1_same_seed_reproducible():
    """acceptance 1: same-seed sweep twice -> identical results."""
    conds = [{"x": i} for i in range(8)]
    r1 = run_sweep(conds, run_rng, seed=42, parallel=1)
    r2 = run_sweep(conds, run_rng, seed=42, parallel=1)
    assert [r["noise"] for r in r1.rows] == [r["noise"] for r in r2.rows]


def test_acceptance2_parallelism_invariant():
    """acceptance 2: parallel in {1,4,8} -> identical results."""
    conds = [{"x": i} for i in range(16)]
    base = run_sweep(conds, run_rng, seed=7, parallel=1)
    base_noise = [r["noise"] for r in base.rows]
    for p in (4, 8):
        res = run_sweep(conds, run_rng, seed=7, parallel=p)
        # order preserved and values identical regardless of process count
        assert [r["_index"] for r in res.rows] == list(range(16))
        assert [r["noise"] for r in res.rows] == base_noise


def test_acceptance3_mean_time_reported():
    """acceptance 3: mean time per condition is available."""
    conds = [{"x": i} for i in range(5)]
    res = run_sweep(conds, run_rng, seed=0)
    assert res.mean_elapsed() >= 0.0
    assert "mean time/condition" in res.report()


def test_acceptance4_failures_skipped_and_listed():
    """acceptance 4: a failing condition does not abort the sweep; it is listed."""
    conds = [{"x": i} for i in range(6)]     # x==3 raises
    res = run_sweep(conds, run_flaky, seed=0, parallel=1)
    assert len(res.rows) == 6                # completed all
    fails = res.failures()
    assert len(fails) == 1 and fails[0]["x"] == 3
    assert "boom" in fails[0]["_error"]
    assert len(res.ok()) == 5


def test_acceptance5_find_boundary_on_synthetic_data():
    """acceptance 5: find_boundary returns the known threshold."""
    widths = np.round(np.arange(0.40, 1.001, 0.02), 3)
    conds = [{"footprint_width": float(w)} for w in widths]
    res = run_sweep(conds, run_boundary, seed=0)
    b = find_boundary(res.ok(), x="footprint_width",
                      predicate=lambda r: r["min_clearance"] >= 0.15)
    assert b == 0.65   # midpoint of 0.64 (pass) and 0.66 (fail)

    # with a refine callable, bisection sharpens it to the true crossing
    b2 = find_boundary(res.ok(), x="footprint_width",
                       predicate=lambda r: r["min_clearance"] >= 0.15,
                       refine=lambda w: (0.8 - w) >= 0.15, tol=1e-4)
    assert abs(b2 - 0.65) < 1e-3


# --------------------------------------------------------------------------- #
def test_sweep_spec_expands_product():
    import tempfile
    text = """
name: t
parallel: 2
seed: 1
parameters:
  a: {range: [0.0, 1.0], step: 0.5}
  b: {values: [x, y]}
metrics: [m]
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        path = fh.name
    spec = SweepSpec.from_yaml(path)
    conds = spec.conditions()
    assert len(conds) == 3 * 2                # a in {0,0.5,1}, b in {x,y}
    assert conds[0] == {"a": 0.0, "b": "x"}
    assert spec.parallel == 2 and spec.seed == 1


def test_bisect_boundary_direct():
    b = bisect_boundary(lambda w: (0.8 - w) >= 0.15, lo=0.4, hi=1.0, tol=1e-4)
    assert abs(b - 0.65) < 1e-3
