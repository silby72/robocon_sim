"""Build a runnable ``RobotConfig`` from ``config/robot/*``.

The mechanism layer could already derive everything about the configured robot
-- ``build.py`` writes ``config/generated/plant_true.yaml`` and
``jacobian.yaml`` from it -- but nothing *ran* on those. ``RobotSim`` was
constructed straight from ``RobotConfig()``'s library defaults, so the ROS node
simulated an idealised 10 kg robot with four ungeared wheels 0.2 m from its
centre while the GUI edited a 15 kg one with 6:1 gearboxes at 0.62 m. For a
1 m/s forward command those two disagree about motor speed by a factor of 5.9,
almost all of it the missing gear ratio. Every gain tuned against the sim was
tuned against a robot that does not exist.

This is the missing join: ``config/robot/*`` in, ``RobotConfig`` out, no files
on disk required (so it does not depend on ``tools/build_plant.py`` having been
run, and it picks up an edit from the chassis/actuator GUI pages immediately).

What does *not* survive the trip, and why:

* **Per-wheel motor differences.** ``MotorParams`` is one spec shared by all
  motors, so a representative drive wheel is used -- the same one ``build.py``
  names in the generated file's header. Mixed drivetrains are misreported here
  rather than silently averaged.
* **A centre-of-mass offset.** ``RigidBody`` integrates a diagonal mass matrix,
  which cannot express the translation/rotation coupling an off-centre CoM
  produces (``mechanism.jacobian.build_mass_matrix`` does, and is used here for
  the parallel-axis term only). A non-zero offset is reported by
  ``RobotConfigWarnings`` rather than quietly dropped.
* **Drag.** ``chassis.yaml`` has no drag terms at all, so the defaults stand.
  They are a modelling choice, not a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..plant.body import BodyParams
from ..plant.jacobian import JacobianParams
from ..plant.odometry import OdometryParams, OdometrySources
from ..plant.motor import MotorParams
from ..simulator import RobotConfig
from .build import (BuildInputs, _motor_params_map, _resolve_actuator,
                    load_presets,
                    load_inputs)
from .jacobian import (build_drive_jacobian, build_mass_matrix,
                       build_odometry_jacobian, odometry_observability)
from .schema import Chassis


@dataclass
class RobotConfigWarnings:
    """What the translation could not represent. Empty is the happy path."""

    messages: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.messages)

    def __str__(self) -> str:
        return "; ".join(self.messages)


def motor_params_from(inp: BuildInputs) -> MotorParams:
    """The representative drive motor, as the true plant's ``MotorParams``."""
    raw = dict(_motor_params_map(inp))
    fields = {f.name for f in MotorParams.__dataclass_fields__.values()}
    return MotorParams(**{k: v for k, v in raw.items() if k in fields})


def jacobian_params_from(chassis: Chassis) -> JacobianParams:
    """The real wheel layout: measured positions, per-wheel radius and gearing.

    ``build_drive_jacobian`` already maps a body twist to motor-shaft speed
    with the gear ratio and ``reverse`` folded in; the matching contact gain is
    the same signed ``n_i / r_i``, so that dividing one by the other recovers
    the unit drive direction the wrench is built from.
    """
    J = build_drive_jacobian(chassis)
    gain = np.array([(w.gear_ratio / w.radius_m) * (-1.0 if w.reverse else 1.0)
                     for w in chassis.drive_wheels])
    radii = [w.radius_m for w in chassis.drive_wheels]
    offsets = [float(np.hypot(*w.position_m)) for w in chassis.drive_wheels]
    return JacobianParams(
        # kept for reference/reporting only -- drive_matrix is what is used
        wheel_radius_m=float(np.mean(radii)),
        body_radius_m=float(np.mean(offsets)),
        drive_matrix=J.tolist(),
        contact_gain=gain.tolist(),
    )


def body_params_from(chassis: Chassis, warnings: RobotConfigWarnings) -> BodyParams:
    com = chassis.center_of_mass
    defaults = BodyParams()
    cx, cy = com.position_m
    if abs(cx) > 1e-9 or abs(cy) > 1e-9:
        warnings.messages.append(
            f"centre of mass is offset ({cx:+.3f}, {cy:+.3f}) m from base_link; "
            "RigidBody's diagonal mass matrix cannot represent the resulting "
            "translation/rotation coupling, so only the parallel-axis inertia "
            "term is carried over")
    return BodyParams(
        mass_kg=float(com.mass_kg),
        # about base_link, not the CoM: build_mass_matrix applies parallel axis
        inertia_z_kgm2=float(build_mass_matrix(chassis)[2, 2]),
        drag_x_ns_m=defaults.drag_x_ns_m,
        drag_y_ns_m=defaults.drag_y_ns_m,
        drag_theta_nms=defaults.drag_theta_nms,
    )


def odometry_params_from(base_dir: str | Path) -> OdometryParams:
    """The ``odometry:`` block of ``model_error.yaml``, if present.

    Optional and absent-means-ideal, so an existing config keeps its exact
    behaviour; the errors are opt-in, named, and live next to the controller's
    model error because they are the same kind of thing.
    """
    from .yaml_rt import rt_load

    path = Path(base_dir) / "config" / "robot" / "model_error.yaml"
    if not path.exists():
        return OdometryParams()
    block = (rt_load(path) or {}).get("odometry") or {}
    ratio = block.get("wheel_radius_ratio", 1.0)
    if isinstance(ratio, (list, tuple)):
        ratio = [float(v) for v in ratio]
    else:
        ratio = float(ratio)
    dw = block.get("dead_wheel_radius_ratio", 1.0)
    if isinstance(dw, (list, tuple)):
        dw = [float(v) for v in dw]
    else:
        dw = float(dw)
    return OdometryParams(
        wheel_radius_ratio=ratio,
        track_ratio=float(block.get("track_ratio", 1.0)),
        slip_ratio=float(block.get("slip_ratio", 1.0)),
        slip_noise_std=float(block.get("slip_noise_std", 0.0)),
        dead_wheel_radius_ratio=dw,
    )


def odometry_sources_from(chassis: Chassis, op: OdometryParams,
                          warnings: RobotConfigWarnings) -> OdometrySources | None:
    """``chassis.yaml``'s ``odometry:`` block as a usable measurement model.

    The schema and the GUI have described dead wheels and optical sensors since
    Phase B; nothing consumed them, so a team could lay out a full odometry pod
    in the editor and the simulator would keep dead-reckoning off the drive
    wheels regardless.

    Two dead wheels cannot observe a 3-DOF twist -- two equations, three
    unknowns -- so a gyro row is added when the units alone are short, which is
    the standard build (two dead wheels + IMU). If it is still unobservable or
    badly conditioned, this returns None and says why: falling back to the
    drive wheels is worse hardware but a defined answer, whereas inverting a
    rank-deficient system is a pose that looks plausible and is wrong in a
    direction nobody chose.
    """
    if not chassis.odometry:
        return None
    matrix, ids = build_odometry_jacobian(chassis, with_gyro=False)
    uses_gyro = False
    ok, cond = odometry_observability(matrix)
    if not ok:
        matrix, ids = build_odometry_jacobian(chassis, with_gyro=True)
        uses_gyro = True
        ok, cond = odometry_observability(matrix)
    if not ok:
        warnings.messages.append(
            f"odometry units {[o.id for o in chassis.odometry]} cannot observe "
            f"the body twist even with a gyro (condition number {cond:.3g}); "
            "falling back to drive-wheel dead reckoning")
        return None

    # believed geometry: the dead wheels' own radius error. A row scales as
    # 1/r, so a believed radius k times the true one divides that row by k.
    belief = matrix.copy()
    k = np.atleast_1d(np.asarray(op.dead_wheel_radius_ratio, dtype=float))
    n_unit_rows = matrix.shape[0] - (1 if uses_gyro else 0)
    if k.size == 1:
        k = np.full(n_unit_rows, float(k[0]))
    if k.size != n_unit_rows:
        raise ValueError(f"dead_wheel_radius_ratio has {k.size} entries for "
                         f"{n_unit_rows} measurement rows")
    belief[:n_unit_rows] /= k[:, None]
    return OdometrySources(matrix=matrix, belief=belief, ids=tuple(ids),
                           uses_gyro=uses_gyro)


def available_motor_presets(base_dir: str | Path = ".") -> list[str]:
    """Preset names under ``config/presets/motors/``, for a UI to offer."""
    return sorted(load_presets(Path(base_dir) / "config" / "presets" / "motors"))


def drivetrain_preset(base_dir: str | Path = ".") -> str | None:
    """The preset every drive wheel currently uses, or None if they differ."""
    inp = load_inputs(base_dir)
    presets = {inp.actuators.actuators[w.actuator_ref].preset
               for w in inp.chassis.drive_wheels}
    return presets.pop() if len(presets) == 1 else None


def _swap_drivetrain_preset(inp: BuildInputs, preset: str) -> None:
    """Point every drive wheel's actuator at ``preset``, in memory only.

    Deliberately not written back to ``config/robot/actuators.yaml``: swapping
    the motor is something you do to *ask a question* ("would AK40s climb this
    ramp?"), and a dashboard click should not rewrite the file the GUI owns.
    Per-unit overrides are kept -- they describe the physical unit, not the
    model -- so a measured inertia survives the swap and keeps applying.
    """
    if preset not in inp.presets:
        raise KeyError(f"unknown motor preset {preset!r}; "
                       f"have {sorted(inp.presets)}")
    for w in inp.chassis.drive_wheels:
        inp.actuators.actuators[w.actuator_ref].preset = preset
        inp.actuators.actuators[w.actuator_ref].datasheet = None


def robot_config_from_dir(base_dir: str | Path = ".", *, clock=None,
                          motor_preset: str | None = None
                          ) -> tuple[RobotConfig, RobotConfigWarnings]:
    """``config/robot/*`` under ``base_dir`` -> a ``RobotConfig`` to simulate.

    ``motor_preset`` substitutes that drive motor for whatever the config
    names, without touching the files -- see ``_swap_drivetrain_preset``.
    """
    inp = load_inputs(base_dir)
    warnings = RobotConfigWarnings()
    if motor_preset is not None:
        _swap_drivetrain_preset(inp, motor_preset)
    cfg = RobotConfig(
        motor=motor_params_from(inp),
        jacobian=jacobian_params_from(inp.chassis),
        body=body_params_from(inp.chassis, warnings),
        odometry=odometry_params_from(base_dir),
        # Built from real datasheets, so hold it to them: without this the
        # drivetrain has no top speed and no stall torque, and swapping the
        # motor changes nothing that moves.
        enforce_motor_envelope=True,
    )
    if clock is not None:
        cfg.clock = clock
    src = odometry_sources_from(inp.chassis, cfg.odometry, warnings)
    if src is not None:
        cfg.odometry = replace(cfg.odometry, sources=src)
    _warn_if_drivetrain_is_mixed(inp, warnings)
    return cfg, warnings


def _warn_if_drivetrain_is_mixed(inp: BuildInputs, warnings: RobotConfigWarnings
                                 ) -> None:
    """Warn only about a drivetrain ``MotorParams`` genuinely cannot express.

    Four wheels naming four *actuator instances* is the normal case -- one
    entry per motor, all pointing at the same preset -- so comparing reference
    names would cry wolf on every well-formed chassis. What matters is whether
    they resolve to the same motor and the same gearing.
    """
    specs = set()
    for w in inp.chassis.drive_wheels:
        datasheet, physical, overrides = _resolve_actuator(inp, w.actuator_ref)
        specs.add((repr(datasheet), repr(physical), repr(sorted(overrides.items())),
                   w.gear_ratio, w.radius_m))
    if len(specs) > 1:
        warnings.messages.append(
            f"the {len(inp.chassis.drive_wheels)} wheels resolve to "
            f"{len(specs)} different motor/gearing specs, but MotorParams is "
            "one shared spec; the first wheel's motor stands in for all of them")


def achievable_speed(cfg: RobotConfig, direction=(1.0, 0.0, 0.0),
                     margin: float = 0.85) -> float:
    """Steady-state body speed [m/s] this drivetrain can actually hold.

    Not the no-load speed. At no-load speed the available torque is zero, so
    the robot cannot even overcome its own drag there; the sustainable speed is
    where the wrench the motors can still produce balances the drag::

        tau_avail(w) = tau_stall * (1 - |w| / w_noload)    per wheel
        F_body(v)    = A^T (gain * tau_avail(J v))         through the jacobian
        solve  F_body(v) . dir  =  drag . v

    This is the number a trajectory profile's ``v_max`` has to respect. It was
    hard-coded at 1.2 m/s while the configured M3508 drivetrain could hold
    about 0.6: every planned trajectory asked for twice the speed the robot
    had, the follower fell permanently behind, and the run was abandoned
    metres short. ``margin`` keeps a little torque in hand for accelerating
    and climbing rather than sitting exactly at the balance point.
    """
    from ..plant.jacobian import JacobianLayer

    jac = JacobianLayer(cfg.jacobian)
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d[:2])), 1e-9)
    drag = np.array([cfg.body.drag_x_ns_m, cfg.body.drag_y_ns_m,
                     cfg.body.drag_theta_nms])
    p = cfg.motor

    # Which way each wheel must turn for this motion, taken at unit speed:
    # reading it off omega at v = 0 is degenerate (every wheel reads zero, so
    # every wheel gets driven the same way, which spins an omni instead of
    # translating it -- the bisection then sees no surplus anywhere).
    omega_unit = jac.wheel_velocity_from_body(d)
    spin = np.sign(omega_unit)
    drag_along = float(drag[0] * d[0] ** 2 + drag[1] * d[1] ** 2)

    def surplus(v: float) -> float:
        avail = p.torque_max_nm * np.clip(
            1.0 - np.abs(omega_unit * v) / max(p.omega_max_rad_s, 1e-9), 0.0, 1.0)
        wrench = jac.body_force_from_wheel(
            jac.wheel_force_from_torque(spin * avail))
        return float(wrench[:2] @ d[:2] - drag_along * v)

    per_unit = float(np.abs(omega_unit).max())
    hi = p.omega_max_rad_s / max(per_unit, 1e-9)
    if surplus(0.0) <= 0.0:
        return 0.0
    lo = 0.0
    for _ in range(60):                       # bisection; surplus is decreasing
        mid = 0.5 * (lo + hi)
        if surplus(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return margin * lo


def uphill_push(cfg: RobotConfig, speed: float = 0.0,
                direction=(1.0, 0.0, 0.0)) -> float:
    """Horizontal force [N] the drivetrain can produce while moving at ``speed``.

    At a standstill this is the stall-torque figure; it is not the number that
    decides whether a robot *climbs*. Available torque falls off with speed, so
    a motor that can hold a slope from rest may be unable to make any progress
    up it -- which is exactly what a weak drivetrain does on the 2027 ramp:
    holds at the foot, oscillates, never gains height. Reporting only the
    standstill figure said "can hold 45 deg" about a robot visibly stuck at the
    bottom of a 30 degree slope.
    """
    from ..plant.jacobian import JacobianLayer

    jac = JacobianLayer(cfg.jacobian)
    d = np.asarray(direction, dtype=float)
    omega_unit = jac.wheel_velocity_from_body(d)
    p = cfg.motor
    avail = p.torque_max_nm * np.clip(
        1.0 - np.abs(omega_unit * speed) / max(p.omega_max_rad_s, 1e-9), 0.0, 1.0)
    wrench = jac.body_force_from_wheel(
        jac.wheel_force_from_torque(np.sign(omega_unit) * avail))
    return float(wrench[:2] @ d[:2])


def climb_limits(cfg: RobotConfig, climb_speed: float = 0.1) -> dict:
    """``{hold_deg, climb_deg, push_hold_n, push_climb_n}`` for a report."""
    from ..field.slope import max_climb_angle_rad

    m = cfg.body.mass_kg
    hold = uphill_push(cfg, 0.0)
    climb = uphill_push(cfg, climb_speed)
    return {
        "hold_deg": float(np.degrees(max_climb_angle_rad(hold, m))),
        "climb_deg": float(np.degrees(max_climb_angle_rad(climb, m))),
        "climb_speed": climb_speed,
        "push_hold_n": hold,
        "push_climb_n": climb,
    }


def describe(cfg: RobotConfig) -> str:
    """One line for a log: what is actually going to be simulated."""
    j = cfg.jacobian
    n = len(j.drive_matrix) if j.drive_matrix is not None else 4
    return (f"{n} wheels, {cfg.body.mass_kg:.1f} kg, "
            f"Iz {cfg.body.inertia_z_kgm2:.3f} kg m^2, "
            f"Kt {cfg.motor.torque_constant_nm_a:.3f} Nm/A, "
            f"R {cfg.motor.resistance_ohm:.2f} ohm")
