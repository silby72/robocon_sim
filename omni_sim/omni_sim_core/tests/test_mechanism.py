"""Phase A tests (spec section 7): mechanism parameter layer + control-rate.

These require ``ruamel.yaml`` (a core dependency). They do NOT require PySide6:
test_gui_independence asserts the mechanism layer and a sim run work with the GUI
toolkit unavailable.
"""
import builtins
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ruamel.yaml is a declared core dependency (installed by setup_env). If a
# minimal environment lacks it, skip these mechanism tests rather than erroring
# on collection -- the base simulator and its 18 tests do not need ruamel.
pytest.importorskip("ruamel.yaml")

from omni_sim_core.mechanism.yaml_rt import rt_load, rt_dumps
from omni_sim_core.mechanism.schema import (
    Chassis, Footprint, CenterOfMass, DriveWheel, Datasheet, Physical,
    MotorPreset)
from omni_sim_core.mechanism.derive import derive_motor
from omni_sim_core.mechanism.jacobian import (
    build_drive_jacobian, DriveJacobian, SingularConfigError)

REPO = Path(__file__).resolve().parents[2]   # .../omni_sim
CONFIG = REPO / "config"


# 1 -- round-trip preserves bytes AND comments ------------------------------
def test_chassis_roundtrip_is_byte_identical():
    path = CONFIG / "robot" / "chassis.yaml"
    original = path.read_text(encoding="utf-8")
    doc = rt_load(path)
    Chassis.from_doc(doc)                     # parse into dataclass (validate)
    assert rt_dumps(doc) == original, "chassis.yaml did not round-trip byte-identically"


# 2 -- M3508 derived constants match the datasheet within 5% ----------------
def test_m3508_derivation_matches_datasheet():
    preset = MotorPreset.from_doc(
        rt_load(CONFIG / "presets" / "motors" / "m3508_c620.yaml"), "m3508_c620")
    d = derive_motor(preset.datasheet, preset.physical,
                     gear_ratio=6.0, load_inertia_wheel_side_kgm2=1e-3)

    # Kt from KV should agree with Kt implied by stall torque / stall current.
    # (DJI M3508+C620, ideal back-emf model: Kt ~ 0.477 Nm/A.)
    kt_from_stall = preset.datasheet.stall_torque_nm / (
        preset.datasheet.stall_current_a - preset.datasheet.no_load_current_a)
    assert abs(d.torque_constant_nm_a - kt_from_stall) / kt_from_stall < 0.05

    # No-load speed from KV*V should match the datasheet's stated no-load speed.
    # (Published M3508 no-load output speed ~469-482 rpm; here 480 rpm.)
    omega_rpm = d.no_load_speed_rad_s * 60.0 / (2 * math.pi)
    assert abs(omega_rpm - preset.datasheet.no_load_speed_rpm) / \
        preset.datasheet.no_load_speed_rpm < 0.05


# helpers for jacobian tests -------------------------------------------------
def _square_omni(L=0.4, r=0.05, gear=1.0) -> Chassis:
    """4-wheel omni, wheels at (+/-L/2, +/-L/2), tangential drive axes."""
    h = L / 2
    wheels = [
        DriveWheel("fl", (h, h), math.radians(135), r, gear, "a", False),
        DriveWheel("fr", (h, -h), math.radians(45), r, gear, "a", False),
        DriveWheel("rl", (-h, h), math.radians(225), r, gear, "a", False),
        DriveWheel("rr", (-h, -h), math.radians(315), r, gear, "a", False),
    ]
    return Chassis(1, Footprint("square", L), CenterOfMass((0, 0), 10.0, 0.3), wheels)


# 3 -- jacobian equals the hand-computed analytic solution ------------------
def test_jacobian_matches_analytic_square_omni():
    J = build_drive_jacobian(_square_omni(L=0.4, r=0.05, gear=1.0))
    s = math.sqrt(2) / 2          # cos/sin of 45 deg
    g = 1.0 / 0.05               # gear/radius = 1/0.05 = 20
    # rows: gain * [cos, sin, x*sin - y*cos]; the wz column is +0.4*s for all.
    expected = g * np.array([
        [-s,  s, 0.4 * s],
        [ s,  s, 0.4 * s],
        [-s, -s, 0.4 * s],
        [ s, -s, 0.4 * s],
    ])
    assert np.allclose(J, expected, atol=1e-9), f"\n{J}\n!=\n{expected}"


# 4 -- odometry round-trip is identity --------------------------------------
def test_jacobian_body_wheel_body_roundtrip():
    dj = DriveJacobian(_square_omni())
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.standard_normal(3)
        w = dj.wheel_speeds_from_body(v)
        v2 = dj.body_from_wheel_speeds(w)
        assert np.allclose(v, v2, atol=1e-9)


# 5 -- singular (parallel drive axes) is detected ---------------------------
def test_parallel_drive_axes_raise():
    # every wheel drives along +x -> vy is unobservable, rank 2 < 3.
    wheels = [DriveWheel(f"w{i}", (0.2 * i - 0.3, 0.1 * i), 0.0, 0.05, 1.0, "a")
              for i in range(4)]
    chassis = Chassis(1, Footprint("square", 0.6),
                      CenterOfMass((0, 0), 10.0, 0.3), wheels)
    with pytest.raises(SingularConfigError):
        DriveJacobian(chassis)


# 6 -- reflected inertia scales as 1/n^2 ------------------------------------
def test_gear_ratio_reflects_load_as_inverse_square():
    ds = Datasheet(rated_voltage_v=24, kv_rpm_per_v=100, stall_torque_nm=2.0,
                   stall_current_a=10, no_load_speed_rpm=2000, no_load_current_a=0.5,
                   encoder_cpr=8192, rotor_inertia_kgm2=1e-4)  # J known -> no estimate
    phys = Physical()
    load = 9.0e-3
    d1 = derive_motor(ds, phys, gear_ratio=3.0, load_inertia_wheel_side_kgm2=load)
    d2 = derive_motor(ds, phys, gear_ratio=6.0, load_inertia_wheel_side_kgm2=load)
    # doubling the gear ratio quarters the load's reflected contribution
    assert d2.load_contribution_kgm2 == pytest.approx(d1.load_contribution_kgm2 / 4.0)
    # rotor inertia is unchanged; only the load contribution shrinks
    assert d1.rotor_inertia_kgm2 == pytest.approx(d2.rotor_inertia_kgm2)


# 7 -- core works with the GUI toolkit unavailable --------------------------
def test_gui_independence(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "PySide6" or name.startswith("PySide6."):
            raise ImportError("PySide6 blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    # mechanism layer + a headless sim must still work
    import importlib
    import omni_sim_core.mechanism.build as build_mod
    importlib.reload(build_mod)
    from omni_sim_core.simulator import MotorControlSim
    from omni_sim_core.plant.motor import MotorParams
    from omni_sim_core.plant.nominal import nominal_from_true
    from omni_sim_core.control.pid import PIDParams
    from omni_sim_core.control.dob import DOBParams
    from omni_sim_core.clock import ClockConfig
    tp = MotorParams(n_motors=1)
    sim = MotorControlSim(tp, nominal_from_true(tp), PIDParams(kp=0.02),
                          DOBParams(enabled=True), ClockConfig(), duration_s=0.05)
    log = sim.run()
    assert np.isfinite(log["omega_true"]).all()
    assert "PySide6" not in sys.modules


# 8 -- legacy scenarios still produce the known numeric result --------------
def test_backward_compatible_scenario():
    from omni_sim_core.config import load_scenario
    from omni_sim_core.run import run_scenario
    scen = load_scenario(CONFIG / "scenarios" / "dob_step_load.yaml")
    # legacy scenario references the hand-written config/plant_*.yaml
    assert scen.plant_true == "config/plant_true.yaml"
    log = run_scenario(scen)
    t = log["t"]
    # values pinned from the pre-mechanism-change behaviour (Kp-only, step load
    # 0.5 N*m at t=1.0, Jn=0.5 Jtrue): DOB estimates the load, speed holds ~49.75.
    steady_omega = float(np.mean(log["omega_true"][t > 1.9]))
    d_hat_final = float(log["d_hat"][-1])
    assert steady_omega == pytest.approx(49.75, abs=0.1)
    assert d_hat_final == pytest.approx(0.5, abs=0.02)


# standalone odometry schema (optional, backward compatible) ----------------
def test_odometry_schema_parses_and_validates():
    from omni_sim_core.mechanism.schema import Chassis, SchemaError
    from omni_sim_core.mechanism.yaml_rt import rt_loads
    text = """
schema_version: 1
footprint: {shape: square, size_m: 0.5}
center_of_mass: {position_m: [0.0, 0.0], mass_kg: 10.0, inertia_zz_kgm2: 0.3}
drive_wheels:
  - {id: fl, position_m: [0.2, 0.2], drive_axis_rad: 2.356, radius_m: 0.05, gear_ratio: 6.0, actuator_ref: a}
  - {id: fr, position_m: [0.2, -0.2], drive_axis_rad: 0.785, radius_m: 0.05, gear_ratio: 6.0, actuator_ref: a}
  - {id: rl, position_m: [-0.2, 0.2], drive_axis_rad: 3.927, radius_m: 0.05, gear_ratio: 6.0, actuator_ref: a}
odometry:
  - {id: odo_x, type: dead_wheel, position_m: [0.1, 0.0], measure_axis_rad: 0.0, radius_m: 0.029, encoder_cpr: 4096}
  - {id: opt0, type: optical, position_m: [0.0, 0.0]}
"""
    chassis = Chassis.from_doc(rt_loads(text))
    assert [o.id for o in chassis.odometry] == ["odo_x", "opt0"]
    assert chassis.odometry[0].type == "dead_wheel"
    # a dead_wheel without a radius is invalid
    bad = rt_loads(text.replace("radius_m: 0.029, ", ""))
    with pytest.raises(SchemaError):
        Chassis.from_doc(bad)


def test_chassis_without_odometry_still_valid():
    # backward compatibility: the key is optional
    from omni_sim_core.mechanism.schema import Chassis
    chassis = Chassis.from_doc(rt_load(CONFIG / "robot" / "chassis.yaml"))
    assert chassis.odometry == []


# override routing: datasheet keys re-derive, param keys apply post-derive -----
def test_datasheet_override_rederives_param_override_applies(tmp_path):
    import math
    import shutil
    from omni_sim_core.mechanism.build import build
    from omni_sim_core.mechanism.yaml_rt import rt_load, rt_dump

    shutil.copytree(CONFIG, tmp_path / "config")
    act_path = tmp_path / "config" / "robot" / "actuators.yaml"
    doc = rt_load(act_path)
    ov = doc["actuators"]["m3508_fl"]["overrides"]
    ov["kv_rpm_per_v"] = 40.0          # datasheet key -> re-derive Kt
    # inertia_kgm2 override (param key) is already present in the file
    rt_dump(doc, act_path)

    res = build(tmp_path)
    true = rt_load(res.plant_true)
    # Kt must follow the overridden KV (not the preset's 20 rpm/V)
    assert float(true["torque_constant_nm_a"]) == pytest.approx(60 / (2 * math.pi * 40), rel=1e-6)
    # the param-level inertia override is still applied verbatim
    assert float(true["inertia_kgm2"]) == pytest.approx(5.2e-4)


# control-rate separation (spec section 5) -----------------------------------
def test_control_rate_hz_sets_motor_period():
    from omni_sim_core.clock import ClockConfig, MultiRateClock
    cfg = ClockConfig(dt_sim=1e-4, dt_nav=2e-2, control_rate_hz=500.0)
    assert cfg.dt_motor == pytest.approx(2e-3)   # 500 Hz -> 2 ms
    clk = MultiRateClock(cfg)
    assert clk._motor_stride == 20               # 2 ms / 0.1 ms
    # control fires every 20th integration step (zero-order hold in between)
    fire_steps = []
    for _ in range(41):
        if clk.motor_fires():
            fire_steps.append(clk.step_index)
        clk.advance()
    assert fire_steps == [0, 20, 40]
