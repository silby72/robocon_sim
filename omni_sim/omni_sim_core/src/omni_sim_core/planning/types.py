"""Frozen data types for the planning layer.

``Path`` is geometry only (invariant 8 / §2.8): the moment time enters, it is a
``control.trajectory.Trajectory``, never a ``Path``. Keeping the boundary in the
type system is the same discipline that keeps the true and nominal plants in
separate files -- it makes the wrong thing impossible to express rather than
merely discouraged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Path:
    """A polyline in world metres. No time, no velocity -- geometry only."""

    points: np.ndarray  # (N, 2) [m]

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
        object.__setattr__(self, "points", pts)

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class OrientationProfile:
    """Body heading as a function of normalised arc length s in [0, 1].

    theta is *unwrapped* (continuous, no +-pi jumps) so downstream interpolation
    and rate limiting behave. See §2.7 and P1 acceptance #12.
    """

    s: np.ndarray       # (M,) monotone in [0, 1]
    theta: np.ndarray   # (M,) [rad], unwrapped

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", np.asarray(self.s, dtype=float).ravel())
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=float).ravel())

    def sample(self, s: float) -> float:
        """Interpolate heading at normalised arc length ``s``."""
        return float(np.interp(s, self.s, self.theta))


@dataclass(frozen=True)
class PlanResult:
    """Outcome of a single plan. Failures are values, never exceptions (§2.3)."""

    path: Path | None
    success: bool
    expanded_nodes: int
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @classmethod
    def failed(cls, expanded_nodes: int = 0, warnings: list[str] | None = None,
               elapsed_s: float = 0.0) -> "PlanResult":
        return cls(path=None, success=False, expanded_nodes=expanded_nodes,
                   warnings=list(warnings or []), elapsed_s=elapsed_s)


@dataclass(frozen=True)
class PlanConfig:
    """Planner tuning. SI units (metres, radians). Loaded from YAML elsewhere.

    ``r_circ`` is the robot's circumscribed radius: collision checking during
    search is orientation-independent (§2.5), which is exactly what licences the
    theta/path split (§2.6).
    """

    r_circ: float = 0.30            # circumscribed radius [m] -> hard inflation
    soft_weight: float = 2.0        # w in c = w*exp(-k*(d - r_circ))
    soft_k: float = 4.0             # k [1/m]; larger => cost decays faster
    d_cutoff: float | None = None   # soft-cost cutoff [m]; None => r_circ + 3/k
    max_shift: float = 1.0          # start/goal escape budget [m] (§2.3)
    allow_diagonal_corner_cut: bool = False  # squeeze diagonally past a lethal cell?
    # An enclosed free pocket smaller than this fraction of the map is treated as
    # an obstacle interior (sealed to occupied before the EDT). Large drivable
    # areas and wall-split halves are never sealed. See cost_field.py. 0 disables.
    seal_area_frac: float = 0.05
