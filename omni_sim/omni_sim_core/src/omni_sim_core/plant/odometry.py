"""What the robot *believes* its wheels did, as distinct from what they did.

``RobotSim.update_odometry`` used the true drive jacobian, so dead reckoning
was exact apart from encoder rounding: 0.01 mm of error after 9 m of driving
and spinning. That is not an odometry model, it is a perfect position sensor
with a quantiser in front of it, and every estimator tested against it was
being handed the answer.

The fix follows the discipline the rest of the plant already uses (see
``plant/nominal.py``): the estimator gets its *own* model, deliberately wrong
in named ways, and the gap between the two is what a localisation filter
exists to close.

Three error sources, chosen because they are the ones that dominate a real
omni's dead reckoning and because each produces a distinct, recognisable
signature:

``wheel_radius_ratio``  believed radius / true radius. Scales estimated
                        translation directly. Per-wheel values are allowed --
                        unequal radii are what make odometry curve away from
                        a straight line rather than just run long or short.
``track_ratio``         believed wheel offsets / true ones. Scales the moment
                        arm, so it shows up almost entirely as a heading-rate
                        error, which then rotates all subsequent translation.
``slip_ratio``          measured wheel rotation per rotation that actually
                        moved the robot. Above 1.0 the wheels turn more than
                        the body travels, the classic drive-wheel slip.

Slip here is an *odometry* error, not a traction model. The wheels are rigidly
coupled to the body (no-slip, plant/jacobian.py) and there is no friction
model to derive real slip from, so this injects the symptom -- encoder counts
that bought no motion -- without pretending to model the cause.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .jacobian import JacobianLayer, JacobianParams


@dataclass
class OdometryParams:
    """All 1.0 / 0.0 means perfect odometry, which is the old behaviour."""

    wheel_radius_ratio: float | Sequence[float] = 1.0
    track_ratio: float = 1.0
    slip_ratio: float = 1.0
    slip_noise_std: float = 0.0          # per-update, relative
    # believed radius / true, for the *standalone* units. Separate from the
    # drive wheels' because they are different hardware: a 24 mm dead wheel's
    # radius error is its own, and it is the error that survives when slip
    # stops mattering.
    dead_wheel_radius_ratio: float | Sequence[float] = 1.0
    # Populated by mechanism.robot_config when chassis.yaml declares any.
    # None = fall back to dead reckoning off the drive wheels.
    sources: "OdometrySources | None" = None

    def is_ideal(self) -> bool:
        r = np.atleast_1d(np.asarray(self.wheel_radius_ratio, dtype=float))
        d = np.atleast_1d(np.asarray(self.dead_wheel_radius_ratio, dtype=float))
        return (np.all(r == 1.0) and np.all(d == 1.0) and self.track_ratio == 1.0
                and self.slip_ratio == 1.0 and self.slip_noise_std == 0.0)


def odometry_jacobian(jp: JacobianParams, op: OdometryParams) -> JacobianLayer:
    """The jacobian the *estimator* uses: the true layout, mis-measured.

    A believed radius ``k`` times the true one divides the whole row by ``k``
    (each count is thought to be worth more distance, so speed is
    overestimated); a believed offset ``m`` times the true one multiplies the
    moment-arm column by ``m`` (the wheels are thought to be further out, so
    the same wheel speeds are read as less rotation).
    """
    if jp.drive_matrix is not None:
        J = np.asarray(jp.drive_matrix, dtype=float).copy()
        k = np.atleast_1d(np.asarray(op.wheel_radius_ratio, dtype=float))
        if k.size == 1:
            k = np.full(J.shape[0], float(k[0]))
        if k.size != J.shape[0]:
            raise ValueError(
                f"wheel_radius_ratio has {k.size} entries for {J.shape[0]} wheels")
        J /= k[:, None]
        J[:, 2] *= op.track_ratio
        gain = np.asarray(jp.contact_gain, dtype=float) / k
        return JacobianLayer(replace(jp, drive_matrix=J.tolist(),
                                     contact_gain=gain.tolist()))

    k = np.atleast_1d(np.asarray(op.wheel_radius_ratio, dtype=float))
    if k.size != 1:
        raise ValueError("per-wheel wheel_radius_ratio needs an explicit "
                         "drive_matrix (the symmetric layout has one radius)")
    return JacobianLayer(replace(
        jp,
        wheel_radius_m=jp.wheel_radius_m * float(k[0]),
        body_radius_m=jp.body_radius_m * op.track_ratio))


def apply_slip(dtheta: np.ndarray, op: OdometryParams,
               rng: np.random.Generator) -> np.ndarray:
    """Encoder increments, inflated by the rotation that produced no motion."""
    if op.slip_ratio == 1.0 and op.slip_noise_std == 0.0:
        return dtheta
    factor = op.slip_ratio
    if op.slip_noise_std > 0.0:
        factor = factor + rng.normal(0.0, op.slip_noise_std, size=dtheta.shape)
    return dtheta * factor


@dataclass
class OdometrySources:
    """Standalone odometry units, as a measurement model the estimator can use.

    Built by ``mechanism.robot_config`` from ``chassis.yaml``'s ``odometry:``
    block, because the plant layer must not import the mechanism layer -- so
    what arrives here is already matrices, not schema objects.

    ``matrix`` is the truth (body twist -> what each unit reads) and ``belief``
    is the same thing mis-measured, exactly as ``odometry_jacobian`` is the
    mis-measured drive jacobian. Two matrices, one discipline.

    The point of these units is that they are *not driven*: a passive wheel has
    no traction to lose, so drive-wheel slip does not appear in their readings.
    That is why ``update_odometry`` does not apply ``slip_ratio`` to them, and
    it is the whole reason a team bolts them on.
    """

    matrix: np.ndarray                  # (n_rows, 3) true
    belief: np.ndarray                  # (n_rows, 3) believed
    ids: tuple[str, ...] = ()
    uses_gyro: bool = False

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    def readings_from_twist(self, twist: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(twist, dtype=float)

    def twist_from_readings(self, readings: np.ndarray) -> np.ndarray:
        """Least squares, so an over-determined set (3+ dead wheels) averages
        rather than picking three arbitrary rows."""
        sol, *_ = np.linalg.lstsq(self.belief, np.asarray(readings, dtype=float),
                                  rcond=None)
        return sol
