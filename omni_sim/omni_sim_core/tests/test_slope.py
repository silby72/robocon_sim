"""Ramps as inclines.

Before this a connector was a flat rectangle: the robot crossed a 600 mm rise
with nothing opposing it, so a drivetrain that could not physically get up the
ramp behaved exactly like one that could.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.field.slope import G, SlopeField, max_climb_angle_rad
from omni_sim_core.field.spec import FieldSpec

FIELD = Path(__file__).resolve().parents[2] / "config" / "field" / "robocon2027.yaml"

CENTER_X = 5.5

pytestmark = pytest.mark.skipif(not FIELD.exists(), reason="repo config/ not available")


@pytest.fixture(scope="module")
def slopes():
    return SlopeField.from_spec(FieldSpec.from_yaml(FIELD))


def test_both_ramps_climb_north_alongside_their_own_slab_edge(slopes):
    """The climb direction is *declared*, not inferred. Inferring it from
    "which axis leaves the slab" read the ramp's 1000 mm width as its run and
    reported 30 deg for a 10 deg ramp -- and a 3500 mm run in +x does not even
    fit between the field edge and the slab.

    Both teams' ramps run north; they mirror in *position*, not in climb
    direction, because each hugs its own side of the slab."""
    red = slopes.at(2.0, 4.5)
    blue = slopes.at(9.0, 4.5)
    assert red.name == "ramp_red" and blue.name == "ramp_blue"
    assert red.uphill() == (0.0, 1.0)
    assert blue.uphill() == (0.0, 1.0)
    assert red.rect[0] < CENTER_X < blue.rect[0]


def test_height_runs_from_the_ground_to_the_slab(slopes):
    red = slopes.at(2.0, 4.5)
    assert red.height_at(2.0, 3.0) == pytest.approx(0.0)
    assert red.height_at(2.0, 6.448) == pytest.approx(0.6)
    assert red.height_at(2.0, 4.724) == pytest.approx(0.3, abs=1e-3)   # halfway
    # and it is clamped, not extrapolated, at the edges
    assert red.height_at(2.0, 0.0) == pytest.approx(0.0)


def test_gravity_points_downhill_and_scales_with_the_angle(slopes):
    red = slopes.at(2.0, 4.5)
    w = slopes.gravity_wrench_body(2.0, 4.5, 0.0, 15.0)
    expected = 15.0 * G * math.sin(red.angle_rad) * math.cos(red.angle_rad)
    assert w[1] == pytest.approx(-expected)    # downhill is -y (the ramp climbs +y)
    assert w[0] == pytest.approx(0.0)
    assert w[2] == pytest.approx(0.0)

    # rotating the body rotates the force into its frame, same magnitude
    w90 = slopes.gravity_wrench_body(2.0, 4.5, math.pi / 2, 15.0)
    assert np.linalg.norm(w90[:2]) == pytest.approx(expected)
    assert w90[0] == pytest.approx(-expected)


def test_off_a_ramp_there_is_no_slope_force(slopes):
    assert slopes.at(5.5, 5.5) is None
    assert slopes.gravity_wrench_body(5.5, 5.5, 0.0, 15.0) == pytest.approx(np.zeros(3))
    assert slopes.at(2.0, 1.0) is None          # south of the ramp's foot


def test_the_ramp_is_the_rulebook_gradient(slopes):
    """[R] 3500 mm on the slope, 600 mm of rise -> asin(600/3500) = 9.9 deg.

    Pinned so a corrected dimension shows up as a failing test rather than a
    silent change. The previous value here was 30.3 deg, which came from
    reading the ramp's *width* as its run."""
    for s in slopes.slopes:
        assert s.rise == pytest.approx(0.6)
        assert s.angle_deg == pytest.approx(9.9, abs=0.15)
        assert math.hypot(s.run, s.rise) == pytest.approx(3.5, abs=0.01)


def test_climb_angle_needs_more_force_for_a_heavier_robot():
    light = max_climb_angle_rad(50.0, 10.0)
    heavy = max_climb_angle_rad(50.0, 20.0)
    assert light > heavy
    # the balance point: F = m g sin(a) cos(a) must hold at the answer
    m, f = 15.0, 60.0
    a = max_climb_angle_rad(f, m)
    assert m * G * math.sin(a) * math.cos(a) == pytest.approx(f)


def test_a_planar_model_saturates_at_45_degrees():
    """Honest limit, not a robot fact: a *horizontal* push gains nothing past
    45 deg. A real machine tips or loses traction long before."""
    assert max_climb_angle_rad(1e6, 15.0) == pytest.approx(math.pi / 4)


def test_the_simulator_actually_feels_the_ramp(slopes):
    """An unpowered robot placed on the ramp must roll back down."""
    from omni_sim_core.simulator import RobotConfig, RobotSim

    sim = RobotSim(RobotConfig(), seed=0)
    sim.reset(pose=np.array([2.0, 4.5, 0.0]))
    for _ in range(3000):
        pose = sim.body.pose
        w = slopes.gravity_wrench_body(float(pose[0]), float(pose[1]),
                                       float(pose[2]), sim.cfg.body.mass_kg)
        sim.step(np.zeros(sim.jac.n_wheels), w)
    assert sim.body.pose[1] < 4.49, "the robot did not roll back down the ramp"
