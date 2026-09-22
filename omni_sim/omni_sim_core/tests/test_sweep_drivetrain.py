"""The drivetrain selection sweep.

The sweep's whole value is that it answers questions a kinematic simulator
cannot -- whether a given motor at a given reduction can actually reach a
speed, climb a slope, or hold a line when the load changes. That only holds if
each swept parameter genuinely reaches the model, so these are mostly
sensitivity checks of the same kind as ``test_wiring_audit``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (ROOT / "config" / "robot" / "chassis.yaml").exists(),
    reason="repo config/ not available")


@pytest.fixture(scope="module")
def dt():
    from omni_sim_core.sweep import drivetrain
    return drivetrain


def test_each_swept_parameter_reaches_the_model(dt):
    base = dt.build_variant(ROOT, 2.0, 0.050, 15.0)
    assert dt.build_variant(ROOT, 4.0, 0.050, 15.0).jacobian.drive_matrix \
        != base.jacobian.drive_matrix
    assert dt.build_variant(ROOT, 2.0, 0.075, 15.0).jacobian.drive_matrix \
        != base.jacobian.drive_matrix
    assert dt.build_variant(ROOT, 2.0, 0.050, 25.0).body.mass_kg != base.body.mass_kg


def test_gearing_and_radius_enter_as_the_ratio_n_over_r(dt):
    """Both act through ``n / r`` -- doubling the reduction and doubling the
    wheel cancel exactly. If they ever stop doing that, one of them is being
    applied in the wrong place."""
    a = np.asarray(dt.build_variant(ROOT, 2.0, 0.050, 15.0).jacobian.drive_matrix)
    b = np.asarray(dt.build_variant(ROOT, 4.0, 0.100, 15.0).jacobian.drive_matrix)
    assert a == pytest.approx(b)


def test_the_reflected_inertia_scales_as_one_over_n_squared(dt):
    """What resists a wheel torque is the body through the jacobian, and the
    jacobian scales with n -- so the reflected inertia goes as 1/n^2. This is
    the number the DOB's nominal model needs; the motor's rotor inertia is
    pinned by an actuator override and does not move with gearing at all."""
    j1, _ = dt.reflected_wheel_dynamics(dt.build_variant(ROOT, 1.0, 0.050, 15.0))
    j3, _ = dt.reflected_wheel_dynamics(dt.build_variant(ROOT, 3.0, 0.050, 15.0))
    assert j1 / j3 == pytest.approx(9.0, rel=0.02)


def test_reflected_inertia_is_not_the_rotor_inertia(dt):
    cfg = dt.build_variant(ROOT, 1.0, 0.050, 15.0)
    jn, _ = dt.reflected_wheel_dynamics(cfg)
    assert jn > 10 * cfg.motor.inertia_kgm2


def test_more_reduction_costs_speed_and_buys_torque(dt):
    from omni_sim_core.mechanism.robot_config import achievable_speed, climb_limits

    slow = dt.build_variant(ROOT, 6.0, 0.050, 15.0)
    fast = dt.build_variant(ROOT, 1.0, 0.050, 15.0)
    assert achievable_speed(fast, (1, 1, 0)) > 3 * achievable_speed(slow, (1, 1, 0))
    assert climb_limits(slow)["push_hold_n"] > climb_limits(fast)["push_hold_n"]


def test_evaluate_is_deterministic_and_complete(dt):
    params = {"gear_ratio": 2.0, "wheel_radius": 0.063, "total_mass_kg": 12.5}
    a = dt.evaluate(params, repo=str(ROOT))
    b = dt.evaluate(params, repo=str(ROOT))
    assert a == b, "the same condition gave two different answers"
    for key in ("v_max_diag", "climb_deg_at_speed", "track_err_dob",
                "track_err_nodob"):
        assert key in a and np.isfinite(a[key])


def test_the_diagonal_is_the_binding_direction(dt):
    """With drive axes at +-45 deg, diagonal motion runs two wheels flat out
    while the other two idle -- so the diagonal is slower than +x, and it is
    the one a requirement has to be written against."""
    row = dt.evaluate({"gear_ratio": 2.0, "wheel_radius": 0.050,
                       "total_mass_kg": 15.0}, repo=str(ROOT))
    assert row["v_max_diag"] < row["v_max_x"]


def test_meets_requires_all_three(dt):
    req = {"v_max_diag_min": 1.0, "ramp_deg": 9.9, "track_err_max": 0.15}
    ok = {"_status": "ok", "v_max_diag": 1.4, "climb_deg_at_speed": 20.0,
          "track_err_dob": 0.05}
    assert dt.meets(ok, req)
    assert not dt.meets({**ok, "v_max_diag": 0.9}, req)
    assert not dt.meets({**ok, "climb_deg_at_speed": 5.0}, req)
    assert not dt.meets({**ok, "track_err_dob": 0.2}, req)
    assert not dt.meets({**ok, "_status": "error"}, req)


def test_a_condition_costs_under_a_second(dt):
    """The sweep's own gate: 1000+ conditions only stay practical if each one
    is cheap."""
    import time

    t0 = time.perf_counter()
    dt.evaluate({"gear_ratio": 3.0, "wheel_radius": 0.050,
                 "total_mass_kg": 15.0}, repo=str(ROOT))
    assert time.perf_counter() - t0 < 1.0
