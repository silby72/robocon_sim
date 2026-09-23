"""Monte Carlo localization for a holonomic omni on a layered field.

The odometry error model (``plant/odometry.py``) made dead reckoning drift for
real -- 1.86 % of path length at 2 % drive slip. Nothing corrected it. This is
the correction: particles propagated by odometry increments and weighted by a
LiDAR likelihood field.

Two things differ from a textbook AMCL and both matter here:

**The motion model is holonomic.** The classic ``sample_motion_model_odometry``
decomposes an increment into rotate-translate-rotate, which is a
differential-drive robot's only way to move and an omni's least natural one. An
omni's odometry increment is a body-frame ``(dx, dy, dtheta)`` with all three
independent, so the noise is applied to those directly. Forcing the
rot-trans-rot form here would inject heading noise proportional to a sideways
translation the robot did without turning at all.

**The map changes under the robot.** Ground, L1 and L2 are three separate
rasters sharing one world frame, so a level change swaps the field rather than
moving the particles. ``set_field`` exists for that, and the filter keeps its
particles across the swap -- the robot did not teleport, only the map it should
be scored against did.

Every random draw goes through an explicitly created ``Generator``, so a run is
reproducible from its seed like everything else in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .likelihood_field import LikelihoodField


@dataclass
class MCLParams:
    n_particles: int = 400
    # Motion noise, as standard deviations per unit of motion. Named for what
    # they couple: trans_trans is the part of a translation that goes astray,
    # rot_trans is the heading error a pure translation causes (wheel radius
    # mismatch does exactly this), and so on.
    sigma_trans_per_m: float = 0.08
    sigma_trans_per_rad: float = 0.02
    sigma_rot_per_rad: float = 0.12
    sigma_rot_per_m: float = 0.03
    # Resample only when the particle set has actually collapsed. Resampling
    # every update throws away diversity for nothing and is the usual cause of
    # premature convergence onto a wrong pose.
    ess_ratio: float = 0.5
    init_pos_std_m: float = 0.10
    init_yaw_std_rad: float = 0.05
    # Injected each resample. A filter with none cannot recover from a
    # divergence, which on this field means a level transition that went wrong.
    random_fraction: float = 0.02
    random_pos_std_m: float = 0.30


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class ParticleFilter:
    """MCL over a likelihood field. Deterministic given ``rng``."""

    def __init__(self, field: LikelihoodField, init_pose, *,
                 params: MCLParams | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.field = field
        self.p = params or MCLParams()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.reset(init_pose)

    # -- state ------------------------------------------------------------
    def reset(self, pose) -> None:
        pose = np.asarray(pose, dtype=float)
        n = self.p.n_particles
        self.particles = np.empty((n, 3))
        self.particles[:, :2] = pose[:2] + self.rng.normal(
            0.0, self.p.init_pos_std_m, size=(n, 2))
        self.particles[:, 2] = pose[2] + self.rng.normal(
            0.0, self.p.init_yaw_std_rad, size=n)
        self.weights = np.full(n, 1.0 / n)
        self._resamples = 0

    def set_field(self, field: LikelihoodField) -> None:
        """Score against a different map from now on, keeping the particles."""
        self.field = field

    # -- prediction -------------------------------------------------------
    def predict(self, d_body) -> None:
        """Propagate by a body-frame odometry increment ``(dx, dy, dtheta)``."""
        dx, dy, dth = (float(v) for v in d_body)
        trans = float(np.hypot(dx, dy))
        if trans < 1e-12 and abs(dth) < 1e-12:
            return
        n = len(self.particles)
        s_t = self.p.sigma_trans_per_m * trans + self.p.sigma_trans_per_rad * abs(dth)
        s_r = self.p.sigma_rot_per_rad * abs(dth) + self.p.sigma_rot_per_m * trans

        noisy = np.empty((n, 3))
        noisy[:, 0] = dx + self.rng.normal(0.0, s_t, n)
        noisy[:, 1] = dy + self.rng.normal(0.0, s_t, n)
        noisy[:, 2] = dth + self.rng.normal(0.0, s_r, n)

        # each particle applies the increment in *its own* frame
        c, s = np.cos(self.particles[:, 2]), np.sin(self.particles[:, 2])
        self.particles[:, 0] += c * noisy[:, 0] - s * noisy[:, 1]
        self.particles[:, 1] += s * noisy[:, 0] + c * noisy[:, 1]
        self.particles[:, 2] = _wrap(self.particles[:, 2] + noisy[:, 2])

    # -- correction -------------------------------------------------------
    def update(self, angles, ranges, mount=(0.0, 0.0, 0.0)) -> float:
        """Reweight on one scan; resample if the set has collapsed.

        Returns the effective sample size ratio after the update.
        """
        logp = self.field.log_likelihood(self.particles, angles, ranges, mount)
        logp -= logp.max()                      # keep exp() in range
        w = self.weights * np.exp(logp)
        total = w.sum()
        if not np.isfinite(total) or total <= 0.0:
            # every particle is impossible: keep the set rather than dividing
            # by zero, and let the next scan (or the random injection) recover
            self.weights = np.full(len(w), 1.0 / len(w))
            return 1.0
        self.weights = w / total
        ess = self.ess_ratio()
        if ess < self.p.ess_ratio:
            self._resample()
        return ess

    def ess_ratio(self) -> float:
        return float(1.0 / (len(self.weights) * np.sum(self.weights ** 2)))

    def _resample(self) -> None:
        """Low-variance (systematic) resampling, plus a few fresh particles."""
        n = len(self.particles)
        n_rand = int(n * self.p.random_fraction)
        n_keep = n - n_rand

        positions = (self.rng.random() + np.arange(n_keep)) / n_keep
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.clip(idx, 0, n - 1)
        new = np.empty_like(self.particles)
        new[:n_keep] = self.particles[idx]
        if n_rand:
            est = self.estimate()
            new[n_keep:, :2] = est[:2] + self.rng.normal(
                0.0, self.p.random_pos_std_m, size=(n_rand, 2))
            new[n_keep:, 2] = est[2] + self.rng.normal(
                0.0, self.p.init_yaw_std_rad * 4, size=n_rand)
        self.particles = new
        self.weights = np.full(n, 1.0 / n)
        self._resamples += 1

    # -- output -----------------------------------------------------------
    def estimate(self) -> np.ndarray:
        """Weighted mean pose. Heading is averaged as a *direction*, because
        the arithmetic mean of 179 deg and -179 deg is 0, pointing backwards."""
        w = self.weights
        x = float(w @ self.particles[:, 0])
        y = float(w @ self.particles[:, 1])
        th = float(np.arctan2(w @ np.sin(self.particles[:, 2]),
                              w @ np.cos(self.particles[:, 2])))
        return np.array([x, y, th])

    def spread(self) -> float:
        """Weighted position standard deviation [m] -- the filter's own
        confidence, and the thing to watch rather than the estimate alone."""
        est = self.estimate()
        d2 = ((self.particles[:, 0] - est[0]) ** 2
              + (self.particles[:, 1] - est[1]) ** 2)
        return float(np.sqrt(self.weights @ d2))

    @property
    def n_resamples(self) -> int:
        return self._resamples
