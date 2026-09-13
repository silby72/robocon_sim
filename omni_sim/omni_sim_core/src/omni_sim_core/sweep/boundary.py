"""Extract a threshold from sweep results (依頼 §5.2).

The point of the sweep is not "0.70 passes, 1.00 fails" but "the boundary is
0.68 m". ``find_boundary`` locates the transition of a predicate along one axis
(coarse: midpoint of the bracketing samples); given a ``refine`` callable it
bisects between the bracket to any tolerance for a sharp number.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def find_boundary(rows: list[dict], x: str, predicate: Callable[[dict], bool],
                  refine: Optional[Callable[[float], bool]] = None,
                  tol: float = 1e-3) -> float | None:
    """Return the ``x`` value where ``predicate`` flips, or None if it never does.

    Rows are grouped by ``x`` (a predicate may be evaluated per row and reduced
    with AND, so a given x "passes" only if all its rows pass), then scanned in
    increasing x for the first True/False (or False/True) transition. The coarse
    boundary is the midpoint of the bracketing x's; ``refine`` bisects it.
    """
    # reduce to one (x, pass) per distinct x value (AND over rows at that x)
    by_x: dict[float, bool] = {}
    for r in rows:
        if x not in r:
            continue
        xv = float(r[x])
        p = bool(predicate(r))
        by_x[xv] = p if xv not in by_x else (by_x[xv] and p)
    if len(by_x) < 2:
        return None

    xs = sorted(by_x)
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if by_x[a] != by_x[b]:
            if refine is None:
                return 0.5 * (a + b)
            return _bisect(refine, a, by_x[a], b, tol)
    return None


def _bisect(refine, lo: float, p_lo: bool, hi: float, tol: float) -> float:
    """Bisect [lo, hi] where refine(lo)==p_lo and refine(hi)!=p_lo."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if bool(refine(mid)) == p_lo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bisect_boundary(refine: Callable[[float], bool], lo: float, hi: float,
                    tol: float = 1e-3) -> float | None:
    """Pure bisection when you can evaluate the predicate at any x directly."""
    p_lo, p_hi = bool(refine(lo)), bool(refine(hi))
    if p_lo == p_hi:
        return None
    return _bisect(refine, lo, p_lo, hi, tol)
