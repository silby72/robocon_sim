"""Parallel sweep executor (依頼 §5.1).

Each condition gets its own ``numpy.random.Generator`` derived from
``(seed, condition_index)`` via ``SeedSequence`` -- NOT from global state and NOT
from the worker. So the result of a condition depends only on its index and the
seed, never on how many processes ran it: same seed + any ``parallel`` gives
bit-identical results (P4 acceptance #1, #2). This is invariant 7 taken to the
multiprocessing case.

A condition that raises is caught, recorded with ``_status='error'`` and its
message, and the sweep continues; the failures are listed at the end
(acceptance #4).
"""
from __future__ import annotations

import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from .results import SweepResults


def _run_condition(indexed, run_one, seed):
    index, params = indexed
    # per-condition RNG: reproducible and independent of process assignment
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))
    row = dict(params)
    row["_index"] = index
    t0 = time.perf_counter()
    try:
        metrics = run_one(params, rng)
        row.update(metrics)
        row["_status"] = "ok"
        row["_error"] = ""
    except Exception as exc:  # keep going; report at the end
        row["_status"] = "error"
        row["_error"] = f"{type(exc).__name__}: {exc}"
    row["_elapsed_s"] = time.perf_counter() - t0
    return row


def run_sweep(conditions, run_one, seed: int = 0, parallel: int = 1
              ) -> SweepResults:
    """Run ``run_one(params, rng) -> dict`` over ``conditions``.

    ``run_one`` must be a top-level (picklable) function returning a flat dict of
    scalar metrics. Order of results follows the condition order regardless of
    ``parallel``.
    """
    tasks = list(enumerate(conditions))
    worker = partial(_run_condition, run_one=run_one, seed=seed)
    if parallel and parallel > 1:
        with Pool(processes=parallel) as pool:
            rows = pool.map(worker, tasks)
    else:
        rows = [worker(t) for t in tasks]
    return SweepResults(rows=rows)
