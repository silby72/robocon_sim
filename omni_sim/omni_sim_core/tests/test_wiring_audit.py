"""Does the configuration actually reach the simulation?

This project has repeatedly shipped features that were declared and never
wired: ``MotorArray`` built but never integrated, ``/sim/disturbance`` stored
but never applied, ``config/robot/*`` overridden by library defaults (5.9x
wrong about the drivetrain), ``odometry[]`` parsed but never read. Every one
of them failed *silently* -- no exception, no warning, just a simulation of a
robot nobody configured.

So these are not unit tests of any one module. They are the audit: perturb an
input and require the output to move. A parameter that can be changed without
changing the answer is not wired, whatever the code looks like.

Kept fast (1 kHz integration, short runs) so it can be the gate that runs
before anything expensive.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.mechanism.yaml_rt import rt_dump, rt_load
from omni_sim_core.simulator import RobotConfig, RobotSim

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

pytestmark = pytest.mark.skipif(not (CONFIG / "robot" / "chassis.yaml").exists(),
                                reason="repo config/ not available")


def _fast(cfg):
    return replace(cfg, clock=replace(cfg.clock, dt_sim=1e-3))


def _dash(cfg, secs=6.0, cmd=(1.0, 0.0, 0.0), kp=0.5):
    """Run a flat-out dash; return (final body twist, final pose)."""
    cfg = _fast(cfg)
    sim = RobotSim(cfg, seed=0)
    for _ in range(int(secs / cfg.clock.dt_sim)):
        sim.step(sim._wheel_torque_from_cmd_vel(np.array(cmd), kp=kp))
    return sim.body.twist_body().copy(), sim.body.pose.copy()


def _built(tmp_path, mutate=None):
    """A RobotConfig built from a scratch copy of config/, optionally edited."""
    import shutil

    from omni_sim_core.mechanism.robot_config import robot_config_from_dir

    base = tmp_path / "repo"
    if not base.exists():
        shutil.copytree(CONFIG, base / "config")
    if mutate is not None:
        mutate(base / "config")
    cfg, _ = robot_config_from_dir(base)
    return cfg


# --------------------------------------------------------------------------- #
# 1. the motor's torque constant
# --------------------------------------------------------------------------- #
def test_a_stronger_motor_accelerates_harder():
    """``RobotSim`` built a ``MotorArray`` and never integrated it, so the
    torque limits were inert: the commanded torque went straight to the body.

    Compared over the *acceleration* phase, not at the end: terminal speed is
    set by drag (F = D v), so two drivetrains that both reach the commanded
    speed finish identically however different they are on the way there.
    """
    # Both the gain and the horizon are chosen so the *torque limit* is what
    # binds. The library default (5 Nm on a 10 kg body) reaches its drag-
    # limited speed in about 35 ms, so any longer run compares two drivetrains
    # that both finished accelerating and shows nothing.
    base = replace(RobotConfig(enforce_motor_envelope=True),
                   motor=replace(RobotConfig().motor, torque_max_nm=0.05))
    strong = replace(base, motor=replace(base.motor, torque_max_nm=0.20))
    _, p0 = _dash(base, secs=0.6, kp=50.0)
    _, p1 = _dash(strong, secs=0.6, kp=50.0)
    assert p1[0] > p0[0] * 2.0, (
        f"a 4x stronger motor accelerated the same ({p0[0]:.4f} vs {p1[0]:.4f} m)")


def test_the_derived_torque_constant_reaches_the_simulation(tmp_path):
    """Kt is not used by ``RobotSim`` directly -- it takes torque commands --
    but it *derives* the stall torque the envelope clamps to. That is the path
    the audit has to follow, and it runs through the motor preset file."""
    def weaker_magnet(cfgdir):
        p = cfgdir / "presets" / "motors" / "m3508_c620.yaml"
        doc = rt_load(p)
        doc["datasheet"]["kv_rpm_per_v"] = 40.0        # half the Kt, twice the speed
        rt_dump(doc, p)

    base = _built(tmp_path / "kt_a")
    weak = _built(tmp_path / "kt_b", weaker_magnet)
    assert weak.motor.torque_constant_nm_a < base.motor.torque_constant_nm_a
    assert weak.motor.torque_max_nm < base.motor.torque_max_nm
    assert weak.motor.omega_max_rad_s > base.motor.omega_max_rad_s
    _, p0 = _dash(base, secs=0.6, cmd=(0.4, 0.0, 0.0))
    _, p1 = _dash(weak, secs=0.6, cmd=(0.4, 0.0, 0.0))
    assert not np.allclose(p0, p1), "the motor preset never reached the sim"


def test_the_torque_speed_envelope_is_enforced():
    """A motor that cannot spin fast enough must cap the robot's speed."""
    from omni_sim_core.mechanism.robot_config import achievable_speed

    base = RobotConfig(enforce_motor_envelope=True)
    # must bind *below* the drag-limited speed or it caps nothing: the default
    # body tops out near 1 m/s on drag alone, at ~14 rad/s of wheel speed.
    slow = replace(base, motor=replace(base.motor, omega_max_rad_s=8.0))
    v_pred = achievable_speed(slow, (1.0, 0.0, 0.0), margin=1.0)
    assert v_pred < 0.9, "the test motor does not actually limit anything"
    v, _ = _dash(slow, secs=10.0, cmd=(v_pred, 0.0, 0.0))
    assert v[0] == pytest.approx(v_pred, rel=0.1)
    assert v[0] < _dash(base, secs=10.0)[0][0]


# --------------------------------------------------------------------------- #
# 2. the gear ratio
# --------------------------------------------------------------------------- #
def test_the_gear_ratio_moves_the_top_speed_as_theory_says(tmp_path):
    """Top speed scales as 1/n: the motor's no-load speed is fixed, and n sits
    between it and the wheel. The 5.9x drivetrain error was exactly this
    factor going missing."""
    from omni_sim_core.mechanism.robot_config import achievable_speed

    def gear(n):
        def _m(cfgdir):
            p = cfgdir / "robot" / "chassis.yaml"
            doc = rt_load(p)
            for w in doc["drive_wheels"]:
                w["gear_ratio"] = n
            rt_dump(doc, p)
        return _m

    speeds = {}
    for n in (2.0, 4.0):
        cfg = _built(tmp_path / f"g{n}", gear(n))
        speeds[n] = achievable_speed(cfg, (1.0, 0.0, 0.0), margin=1.0)
        measured, _ = _dash(cfg, secs=12.0, cmd=(speeds[n], 0.0, 0.0))
        assert measured[0] == pytest.approx(speeds[n], rel=0.1), (
            f"gear {n}: modelled {speeds[n]:.3f} m/s, reached {measured[0]:.3f}")
    # doubling the reduction halves the speed
    assert speeds[2.0] / speeds[4.0] == pytest.approx(2.0, rel=0.05)


def test_the_wheel_radius_moves_the_top_speed(tmp_path):
    from omni_sim_core.mechanism.robot_config import achievable_speed

    def radius(r):
        def _m(cfgdir):
            p = cfgdir / "robot" / "chassis.yaml"
            doc = rt_load(p)
            for w in doc["drive_wheels"]:
                w["radius_m"] = r
            rt_dump(doc, p)
        return _m

    small = achievable_speed(_built(tmp_path / "r1", radius(0.038)), margin=1.0)
    big = achievable_speed(_built(tmp_path / "r2", radius(0.076)), margin=1.0)
    assert big / small == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------------- #
# 3. external disturbance
# --------------------------------------------------------------------------- #
def test_an_external_wrench_actually_disturbs_the_robot():
    """``/sim/disturbance`` was received, stored in ``_ext_wrench`` and never
    passed to ``step()`` -- there was no parameter to pass it to."""
    cfg = _fast(RobotConfig())
    quiet, pushed = RobotSim(cfg, seed=0), RobotSim(cfg, seed=0)
    wrench = np.array([0.0, 40.0, 0.0])        # 40 N sideways
    for _ in range(3000):
        quiet.step(np.zeros(4))
        pushed.step(np.zeros(4), wrench)
    assert quiet.body.pose[:2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert pushed.body.pose[1] > 0.05, "the disturbance did nothing"


def test_the_disturbance_direction_is_respected():
    cfg = _fast(RobotConfig())
    out = {}
    for name, w in (("+y", [0.0, 40.0, 0.0]), ("-y", [0.0, -40.0, 0.0]),
                    ("spin", [0.0, 0.0, 8.0])):
        sim = RobotSim(cfg, seed=0)
        for _ in range(3000):
            sim.step(np.zeros(4), np.array(w))
        out[name] = sim.body.pose.copy()
    assert out["+y"][1] > 0 > out["-y"][1]
    assert out["spin"][2] > 0.05


# --------------------------------------------------------------------------- #
# 4. config/robot/* reaches RobotSim (no silent fallback to library defaults)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,mutate,attr", [
    ("mass", lambda d: d["center_of_mass"].__setitem__("mass_kg", 40.0),
     lambda c: c.body.mass_kg),
    ("inertia", lambda d: d["center_of_mass"].__setitem__("inertia_zz_kgm2", 1.9),
     lambda c: c.body.inertia_z_kgm2),
])
def test_chassis_scalars_reach_the_config(tmp_path, field, mutate, attr):
    def _m(cfgdir):
        p = cfgdir / "robot" / "chassis.yaml"
        doc = rt_load(p)
        mutate(doc)
        rt_dump(doc, p)

    before = attr(_built(tmp_path / f"{field}_a"))
    after = attr(_built(tmp_path / f"{field}_b", _m))
    assert before != after, f"chassis.yaml {field} never reached RobotConfig"


def test_the_built_robot_is_not_the_library_default(tmp_path):
    """The specific failure that started all of this: ``RobotSim`` was
    constructed from ``RobotConfig()`` while the GUI edited config/robot/*."""
    cfg = _built(tmp_path)
    default = RobotConfig()
    assert cfg.body.mass_kg != default.body.mass_kg
    assert cfg.jacobian.drive_matrix is not None, "the real wheel layout is missing"
    j_cfg = np.abs(np.asarray(cfg.jacobian.drive_matrix)[:, 0]).max()
    from omni_sim_core.plant.jacobian import JacobianLayer
    j_def = np.abs(JacobianLayer().wheel_velocity_from_body(
        np.array([1.0, 0.0, 0.0]))).max()
    assert j_cfg / j_def > 3.0, "the drivetrain still looks like the defaults"


def test_a_chassis_edit_changes_how_the_robot_moves(tmp_path):
    """End to end: edit the file, and the trajectory differs. Every layer in
    between has to be wired for this to hold."""
    def heavier(cfgdir):
        p = cfgdir / "robot" / "chassis.yaml"
        doc = rt_load(p)
        doc["center_of_mass"]["mass_kg"] = 40.0
        rt_dump(doc, p)

    light = _built(tmp_path / "light")
    heavy = _built(tmp_path / "heavy", heavier)

    # Measured with a deliberately weak motor. The configured M3508-at-6:1 is
    # so over-geared that it produces ~1600 N for a 15 kg robot -- acceleration
    # is effectively instantaneous and the run is speed-limited, so mass barely
    # shows. That is a real property of this drivetrain (and one the selection
    # sweep exists to fix), but it makes a mass probe blind, so the probe uses
    # an actuator that is torque-limited.
    weak = lambda c: replace(c, motor=replace(c.motor, torque_max_nm=0.08))
    _, p_light = _dash(weak(light), secs=1.0, cmd=(0.3, 0.0, 0.0), kp=50.0)
    _, p_heavy = _dash(weak(heavy), secs=1.0, cmd=(0.3, 0.0, 0.0), kp=50.0)
    assert p_light[0] > p_heavy[0] * 1.5, (
        f"a heavier robot accelerated the same ({p_light[0]:.3f} vs "
        f"{p_heavy[0]:.3f} m)")
