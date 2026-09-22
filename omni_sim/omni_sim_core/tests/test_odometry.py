"""The odometry error model (P5).

Dead reckoning used the *true* drive jacobian, so it was exact apart from
encoder rounding: 0.01 mm of error after 9 m of driving and spinning. Any
estimator tested against that was being handed the answer.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from omni_sim_core.plant.jacobian import JacobianLayer, JacobianParams
from omni_sim_core.plant.odometry import (OdometryParams, apply_slip,
                                          odometry_jacobian)
from omni_sim_core.sensors.encoder import Encoder, EncoderParams
from omni_sim_core.clock import ClockConfig  # noqa: F401
from omni_sim_core.simulator import RobotConfig, RobotSim


def _drive(cfg, secs=20.0, cmd=(0.3, 0.0, 0.4), seed=0):
    """Drive and spin; return (path length, position error, heading error).

    Integrated at 1 kHz rather than the simulator's default 10 kHz: these
    tests assert ratios and growth, not trajectories, and the full rate turned
    the file into a minute of wall clock on its own.
    """
    cfg = replace(cfg, clock=replace(cfg.clock, dt_sim=1e-3))
    sim = RobotSim(cfg, seed=seed)
    enc = Encoder(EncoderParams(), np.random.default_rng(seed))
    rpc = enc._rad_per_count
    dist, prev = 0.0, sim.body.pose[:2].copy()
    for _ in range(int(secs / cfg.clock.dt_sim)):
        sim.step(sim._wheel_torque_from_cmd_vel(np.array(cmd), kp=0.5))
        sim.update_odometry(wheel_angle=rpc * np.round(sim.wheel_angle_true / rpc))
        dist += float(np.linalg.norm(sim.body.pose[:2] - prev))
        prev = sim.body.pose[:2].copy()
    gt, od = sim.body.pose, sim.odom_pose
    return dist, float(np.linalg.norm(gt[:2] - od[:2])), float(gt[2] - od[2])


def test_ideal_params_reproduce_the_old_perfect_odometry():
    """Absent config must change nothing -- the errors are opt-in."""
    assert OdometryParams().is_ideal()
    dist, err, dth = _drive(RobotConfig())
    assert dist > 1.0
    assert err < 1e-3, "ideal odometry should still be essentially exact"


def test_a_believed_radius_scales_the_estimate():
    """Believe the wheels are 10 % bigger and every metre reads as 1.1 m."""
    jp = JacobianParams()
    ideal = JacobianLayer(jp)
    big = odometry_jacobian(jp, OdometryParams(wheel_radius_ratio=1.1))
    omega = ideal.wheel_velocity_from_body(np.array([1.0, 0.0, 0.0]))
    assert big.body_velocity_from_wheel(omega)[0] == pytest.approx(1.1)


def test_a_believed_track_scales_the_heading_rate():
    """Believe the wheels sit further out and the same wheel speeds read as
    less rotation -- the term that dominates real drift."""
    jp = JacobianParams()
    ideal = JacobianLayer(jp)
    wide = odometry_jacobian(jp, OdometryParams(track_ratio=1.1))
    omega = ideal.wheel_velocity_from_body(np.array([0.0, 0.0, 1.0]))
    assert wide.body_velocity_from_wheel(omega)[2] == pytest.approx(1 / 1.1, rel=1e-6)


def test_slip_inflates_the_measured_rotation():
    rng = np.random.default_rng(0)
    d = np.array([1.0, 1.0, 1.0, 1.0])
    assert apply_slip(d, OdometryParams(), rng) is d          # untouched when ideal
    assert apply_slip(d, OdometryParams(slip_ratio=1.02), rng) == pytest.approx(1.02 * d)
    noisy = apply_slip(d, OdometryParams(slip_noise_std=0.1), rng)
    assert not np.allclose(noisy, d) and noisy.shape == d.shape


def test_per_wheel_radius_error_needs_an_explicit_layout():
    """The symmetric layout has exactly one radius, so per-wheel values there
    would be silently ignored -- refuse instead."""
    with pytest.raises(ValueError):
        odometry_jacobian(JacobianParams(),
                          OdometryParams(wheel_radius_ratio=[1.0, 1.01, 1.0, 1.0]))


def test_each_error_source_produces_drift():
    """Every named source must actually move the estimate, or it is decoration."""
    base = RobotConfig()
    _, ideal_err, _ = _drive(base)
    for odo in (OdometryParams(wheel_radius_ratio=1.01),
                OdometryParams(track_ratio=1.01),
                OdometryParams(slip_ratio=1.01)):
        _, err, _ = _drive(replace(base, odometry=odo))
        assert err > 20 * max(ideal_err, 1e-6), f"{odo} produced no drift"


def test_position_drift_grows_with_distance_on_a_straight_run():
    """A scale error accumulates: twice the distance, twice the error.

    Measured on a *straight* run on purpose. While the robot is also spinning,
    a heading error makes the estimate orbit the true pose, so position error
    is bounded rather than growing -- which is real behaviour, and makes a
    straight line the only honest place to assert growth.
    """
    cfg = replace(RobotConfig(), odometry=OdometryParams(wheel_radius_ratio=1.02))
    _, short, _ = _drive(cfg, secs=5.0, cmd=(0.3, 0.0, 0.0))
    _, long, _ = _drive(cfg, secs=10.0, cmd=(0.3, 0.0, 0.0))
    assert long > short * 1.6


def test_heading_drift_grows_with_rotation():
    """The term that ruins a long run: a track error is a heading *rate*
    error, so it integrates without bound while the robot turns."""
    cfg = replace(RobotConfig(), odometry=OdometryParams(track_ratio=1.02))
    _, _, short = _drive(cfg, secs=5.0, cmd=(0.0, 0.0, 0.6))
    _, _, long = _drive(cfg, secs=10.0, cmd=(0.0, 0.0, 0.6))
    assert abs(long) > abs(short) * 1.6


def test_the_true_pose_is_untouched_by_the_odometry_model():
    """The error model must live entirely in the estimate. If it moves the
    robot, it is not an odometry model -- it is a plant bug."""
    a = RobotSim(RobotConfig(), seed=0)
    b = RobotSim(replace(RobotConfig(),
                         odometry=OdometryParams(wheel_radius_ratio=1.2,
                                                 track_ratio=1.2,
                                                 slip_ratio=1.2)), seed=0)
    for _ in range(5000):
        cmd = np.array([0.3, 0.1, 0.2])
        a.step(a._wheel_torque_from_cmd_vel(cmd, kp=0.5))
        b.step(b._wheel_torque_from_cmd_vel(cmd, kp=0.5))
        a.update_odometry()
        b.update_odometry()
    assert a.body.pose == pytest.approx(b.body.pose)
    assert not np.allclose(a.odom_pose, b.odom_pose)


# --------------------------------------------------------------------------- #
# standalone odometry units (dead wheels / optical flow)
# --------------------------------------------------------------------------- #
def _chassis_with(sources):
    from omni_sim_core.mechanism.schema import Chassis
    from omni_sim_core.mechanism.yaml_rt import rt_load
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    ch = Chassis.from_doc(rt_load(root / "config" / "robot" / "chassis.yaml"))
    ch.odometry = sources
    return ch


def _dead_wheel(i, x, y, th, r=0.024):
    from omni_sim_core.mechanism.schema import OdometrySource
    return OdometrySource(id=i, type="dead_wheel", position_m=(x, y),
                          measure_axis_rad=th, radius_m=r, encoder_cpr=4096)


def test_two_dead_wheels_cannot_observe_a_three_dof_twist():
    """Two equations, three unknowns. The standard build adds a gyro for
    exactly this reason, and inverting it anyway would give a pose that looks
    plausible and is wrong in a direction nobody chose."""
    from omni_sim_core.mechanism.jacobian import (build_odometry_jacobian,
                                                  odometry_observability)
    ch = _chassis_with([_dead_wheel("dw_x", 0.0, 0.15, 0.0),
                        _dead_wheel("dw_y", 0.15, 0.0, np.pi / 2)])
    M, _ = build_odometry_jacobian(ch, with_gyro=False)
    assert not odometry_observability(M)[0]
    M, ids = build_odometry_jacobian(ch, with_gyro=True)
    ok, cond = odometry_observability(M)
    assert ok and cond < 1e3 and ids[-1] == "gyro.wz"


def test_three_dead_wheels_need_no_gyro():
    from omni_sim_core.mechanism.jacobian import (build_odometry_jacobian,
                                                  odometry_observability)
    ch = _chassis_with([_dead_wheel("a", 0.0, 0.15, 0.0),
                        _dead_wheel("b", 0.0, -0.15, 0.0),
                        _dead_wheel("c", 0.15, 0.0, np.pi / 2)])
    assert odometry_observability(build_odometry_jacobian(ch)[0])[0]


def test_an_optical_sensor_contributes_two_rows():
    from omni_sim_core.mechanism.jacobian import build_odometry_jacobian
    from omni_sim_core.mechanism.schema import OdometrySource
    ch = _chassis_with([OdometrySource(id="flow", type="optical",
                                       position_m=(0.05, -0.05))])
    M, ids = build_odometry_jacobian(ch, with_gyro=True)
    assert ids == ["flow.vx", "flow.vy", "gyro.wz"]
    # it reads the planar velocity at its own mounting point
    assert M[0] == pytest.approx([1.0, 0.0, 0.05])
    assert M[1] == pytest.approx([0.0, 1.0, 0.05])


def test_an_unobservable_set_falls_back_instead_of_inventing_a_pose():
    from omni_sim_core.mechanism.robot_config import (RobotConfigWarnings,
                                                      odometry_sources_from)
    # two dead wheels on parallel axes: even a gyro cannot fix that
    ch = _chassis_with([_dead_wheel("a", 0.0, 0.15, 0.0),
                        _dead_wheel("b", 0.0, -0.15, 0.0)])
    warnings = RobotConfigWarnings()
    assert odometry_sources_from(ch, OdometryParams(), warnings) is None
    assert warnings.messages and "cannot observe" in warnings.messages[0]


def test_the_pod_is_immune_to_drive_wheel_slip():
    """The entire reason for the hardware: a passive wheel has no traction to
    lose. Drive-wheel dead reckoning degrades with slip; the pod does not."""
    from omni_sim_core.mechanism.robot_config import (RobotConfigWarnings,
                                                      odometry_sources_from)
    ch = _chassis_with([_dead_wheel("dw_x", 0.0, 0.15, 0.0),
                        _dead_wheel("dw_y", 0.15, 0.0, np.pi / 2)])
    src = odometry_sources_from(ch, OdometryParams(), RobotConfigWarnings())
    assert src is not None

    def drift(slip, sources):
        cfg = replace(RobotConfig(),
                      odometry=OdometryParams(slip_ratio=slip, sources=sources))
        return _drive(cfg, secs=10.0)[1]

    assert drift(1.05, None) > 20 * drift(1.0, None), "slip must hurt the wheels"
    assert drift(1.05, src) == pytest.approx(drift(1.0, src), abs=1e-9)


def test_the_pod_still_has_its_own_radius_error():
    """Not magic -- just differently wrong. A dead wheel's own radius error is
    what is left once slip stops mattering."""
    from omni_sim_core.mechanism.robot_config import (RobotConfigWarnings,
                                                      odometry_sources_from)
    ch = _chassis_with([_dead_wheel("dw_x", 0.0, 0.15, 0.0),
                        _dead_wheel("dw_y", 0.15, 0.0, np.pi / 2)])
    exact = odometry_sources_from(ch, OdometryParams(), RobotConfigWarnings())
    off = odometry_sources_from(ch, OdometryParams(dead_wheel_radius_ratio=1.01),
                                RobotConfigWarnings())
    assert np.allclose(exact.belief, exact.matrix)
    assert not np.allclose(off.belief, off.matrix)
    # the gyro row is a rate measurement, not a wheel -- it must not be scaled
    assert off.belief[-1] == pytest.approx(off.matrix[-1])
