"""Derived actuator limits, in a form a Gazebo joint can be given.

Why this exists, in one sentence: the numbers in ``config/generated/plant_*.yaml``
are **motor-shaft** quantities, a Gazebo joint is the **wheel**, and the two
differ by the gear ratio -- which is the 5.9x class of error this project has
already shipped once.

Concretely, for the M3508 at 6:1 the plant file says::

    torque_max_nm:   4.77          # at the motor shaft
    omega_max_rad_s: 50.27         # at the motor shaft

and the wheel joint's limits are::

    effort   4.77 * 6  = 28.6 N m
    velocity 50.27 / 6 = 8.38 rad/s      (0.43 m/s on a 50.8 mm wheel)

Copy the first pair into an SDF ``<limit>`` and the simulated robot is six
times too fast and six times too weak. So this module will not guess: the gear
ratio and wheel radius have to be supplied, or come from a chassis file, and
if neither is available it raises rather than defaulting to 1.0.

A note on Gazebo's ``VelocityControl`` / ``DiffDrive`` plugins: they turn a
velocity command straight into body motion and have no torque concept at all,
so they will happily execute a command no motor could produce. The effort
limit exported here therefore cannot be enforced *by* those plugins -- it is
for whoever generates the commands, and for ``JointController``-style plugins
and SDF ``<limit>`` blocks that do read it. Deciding a drivetrain against a
kinematic plugin is exactly the gap omni_sim covers.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, replace
from pathlib import Path

import yaml


class ExportError(ValueError):
    """Raised rather than exporting a number that would be silently wrong."""


@dataclass(frozen=True)
class WheelLimits:
    """Limits at the wheel joint -- the thing Gazebo actually rotates."""

    gear_ratio: float            # motor:wheel
    wheel_radius_m: float
    effort_limit_nm: float       # tau_stall * n
    velocity_limit_rad_s: float  # omega_noload / n
    # The speed of the wheel's own contact patch -- NOT the robot's speed.
    # On an X-layout omni the drive axes sit at +-45 deg, so a body moving
    # along +x only rolls each wheel at cos(45) of that speed: the body can
    # travel a factor sqrt(2) faster than this number, and on the diagonal
    # it travels exactly this number. Reporting it as "max speed" would be a
    # quiet 1.41x error of the same family as the 6x one above.
    wheel_surface_speed_ms: float
    # carried through unchanged, for anyone who wants to model the motor too
    motor_stall_torque_nm: float
    motor_no_load_rad_s: float
    torque_constant_nm_a: float
    resistance_ohm: float
    # filled in when a chassis is available (the jacobian is needed for them)
    body_speed_axis_ms: float | None = None      # along +x / +y
    body_speed_diagonal_ms: float | None = None  # the binding direction


def wheel_limits(*, motor_stall_torque_nm: float, motor_no_load_rad_s: float,
                 gear_ratio: float, wheel_radius_m: float,
                 torque_constant_nm_a: float = float("nan"),
                 resistance_ohm: float = float("nan")) -> WheelLimits:
    """Motor-shaft limits -> wheel-joint limits. The whole conversion."""
    if gear_ratio <= 0 or wheel_radius_m <= 0:
        raise ExportError("gear_ratio and wheel_radius_m must be > 0 "
                          f"(got {gear_ratio}, {wheel_radius_m})")
    vel = motor_no_load_rad_s / gear_ratio
    return WheelLimits(
        gear_ratio=float(gear_ratio),
        wheel_radius_m=float(wheel_radius_m),
        effort_limit_nm=float(motor_stall_torque_nm * gear_ratio),
        velocity_limit_rad_s=float(vel),
        wheel_surface_speed_ms=float(vel * wheel_radius_m),
        motor_stall_torque_nm=float(motor_stall_torque_nm),
        motor_no_load_rad_s=float(motor_no_load_rad_s),
        torque_constant_nm_a=float(torque_constant_nm_a),
        resistance_ohm=float(resistance_ohm),
    )


def _body_speeds(chassis_yaml: Path, motor_no_load_rad_s: float
                 ) -> tuple[float, float]:
    """``(axis, diagonal)`` body speed at the motor's no-load rate, [m/s].

    Read off the real drive jacobian rather than assumed, because the factor
    between wheel speed and body speed is a property of the wheel layout, not
    a constant: it is sqrt(2) for axes at 45 deg and something else for any
    other arrangement.
    """
    import numpy as np

    from ..mechanism.jacobian import build_drive_jacobian
    from ..mechanism.schema import Chassis
    from ..mechanism.yaml_rt import rt_load

    J = build_drive_jacobian(Chassis.from_doc(rt_load(chassis_yaml)))

    def speed(direction):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d[:2])
        per_unit = float(np.abs(J @ d).max())     # motor rad/s per 1 m/s
        return motor_no_load_rad_s / max(per_unit, 1e-9)

    return speed([1.0, 0.0, 0.0]), speed([1.0, 1.0, 0.0])


def _drivetrain_from_chassis(chassis_yaml: Path) -> tuple[float, float, list[str]]:
    """``(gear_ratio, wheel_radius, wheel_ids)`` -- and refuse a mixed set.

    A single joint limit cannot describe wheels with different gearing, and
    emitting one anyway would be the same silent-averaging failure the export
    exists to prevent.
    """
    from ..mechanism.schema import Chassis
    from ..mechanism.yaml_rt import rt_load

    chassis = Chassis.from_doc(rt_load(chassis_yaml))
    ratios = {w.gear_ratio for w in chassis.drive_wheels}
    radii = {w.radius_m for w in chassis.drive_wheels}
    if len(ratios) != 1 or len(radii) != 1:
        raise ExportError(
            f"{chassis_yaml} has mixed drivetrains (gear ratios {sorted(ratios)}, "
            f"radii {sorted(radii)}); one joint limit cannot describe them. "
            "Export per wheel group, or pass gear_ratio/wheel_radius_m "
            "explicitly for the group you mean.")
    return (ratios.pop(), radii.pop(),
            [w.id for w in chassis.drive_wheels])


def sdf_snippet(lim: WheelLimits, joint_names: list[str]) -> str:
    """An SDF ``<limit>`` block per drive joint, ready to paste."""
    out = []
    for name in joint_names:
        out.append(
            f'<joint name="{name}_joint" type="revolute">\n'
            f'  <axis>\n'
            f'    <limit>\n'
            f'      <effort>{lim.effort_limit_nm:.4f}</effort>'
            f'        <!-- N m at the WHEEL ({lim.motor_stall_torque_nm:.4f}'
            f' at the motor x {lim.gear_ratio:g}) -->\n'
            f'      <velocity>{lim.velocity_limit_rad_s:.4f}</velocity>'
            f'    <!-- rad/s at the WHEEL ({lim.motor_no_load_rad_s:.4f}'
            f' at the motor / {lim.gear_ratio:g}) -->\n'
            f'    </limit>\n'
            f'  </axis>\n'
            f'</joint>')
    return "\n".join(out)


def export_actuator_config(plant_config: Path, output_path: Path, *,
                           chassis_yaml: Path | None = None,
                           gear_ratio: float | None = None,
                           wheel_radius_m: float | None = None,
                           joint_names: list[str] | None = None,
                           body_speeds: tuple[float, float] | None = None,
                           write_sdf: bool = True) -> WheelLimits:
    """Write the wheel-joint limits derived from ``plant_config``.

    ``plant_config`` is a generated ``plant_*.yaml``. It carries motor-shaft
    numbers and **not** the gear ratio -- that appears only in a comment -- so
    the ratio and wheel radius come from ``chassis_yaml`` or from the explicit
    arguments. With neither, this raises: defaulting to 1:1 is precisely the
    mistake the file is meant to prevent.

    Returns the limits, and writes ``output_path`` (plus a ``.sdf`` sibling
    when ``write_sdf``), each stamped with where the numbers came from.
    """
    plant_config, output_path = Path(plant_config), Path(output_path)
    if not plant_config.exists():
        raise ExportError(f"no plant config at {plant_config}")
    plant = yaml.safe_load(plant_config.read_text(encoding="utf-8")) or {}
    for key in ("torque_max_nm", "omega_max_rad_s"):
        if key not in plant:
            raise ExportError(
                f"{plant_config} has no '{key}'. It does not look like a "
                "generated plant file; regenerate with tools/build_plant.py")

    ids = joint_names
    if gear_ratio is None or wheel_radius_m is None:
        if chassis_yaml is None:
            raise ExportError(
                "the gear ratio and wheel radius are not in a plant file -- "
                "they appear only in a comment. Pass chassis_yaml=, or "
                "gear_ratio= and wheel_radius_m=. Refusing to assume 1:1: "
                "that would export motor-shaft numbers as wheel limits, which "
                "is wrong by exactly the gear ratio.")
        n, r, chassis_ids = _drivetrain_from_chassis(Path(chassis_yaml))
        gear_ratio = n if gear_ratio is None else gear_ratio
        wheel_radius_m = r if wheel_radius_m is None else wheel_radius_m
        ids = ids or chassis_ids
    ids = ids or ["wheel"]

    lim = wheel_limits(
        motor_stall_torque_nm=float(plant["torque_max_nm"]),
        motor_no_load_rad_s=float(plant["omega_max_rad_s"]),
        gear_ratio=float(gear_ratio), wheel_radius_m=float(wheel_radius_m),
        torque_constant_nm_a=float(plant.get("torque_constant_nm_a", float("nan"))),
        resistance_ohm=float(plant.get("resistance_ohm", float("nan"))),
    )
    if body_speeds is None and chassis_yaml is not None:
        body_speeds = _body_speeds(Path(chassis_yaml), lim.motor_no_load_rad_s)
    if body_speeds is not None:
        lim = replace(lim, body_speed_axis_ms=float(body_speeds[0]),
                      body_speed_diagonal_ms=float(body_speeds[1]))

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        "# GENERATED by omni_sim_core.export.gazebo_actuator -- do not edit.",
        f"# generated: {stamp}",
        f"# plant:     {plant_config}",
        f"# chassis:   {chassis_yaml if chassis_yaml else '(explicit arguments)'}",
        "#",
        "# ALL LIMITS ARE AT THE WHEEL JOINT, not the motor shaft. The plant",
        f"# file's numbers are motor-side; they differ by the {lim.gear_ratio:g}:1",
        "# reduction. Using them directly makes the robot",
        f"# {lim.gear_ratio:g}x too fast and {lim.gear_ratio:g}x too weak.",
        "#",
        "# Gazebo's VelocityControl / DiffDrive plugins have no torque concept",
        "# and will execute a command no motor could produce; the effort limit",
        "# is for the command generator and for <limit>-reading plugins.",
        "#",
        "# wheel_surface_speed_ms is the wheel's contact patch, NOT the robot.",
        "# On this layout the body moves body_speed_axis_ms along +x/+y and",
        "# body_speed_diagonal_ms on the diagonal; the diagonal is the binding",
        "# one, because two wheels run flat out while the other two idle.",
        "",
    ]
    body = {
        "joint_velocity_limit": round(lim.velocity_limit_rad_s, 6),
        "joint_effort_limit": round(lim.effort_limit_nm, 6),
        "wheel_radius_m": lim.wheel_radius_m,
        # the wheel's contact patch, not the robot -- see the note below
        "wheel_surface_speed_ms": round(lim.wheel_surface_speed_ms, 6),
        "body_speed_axis_ms": (None if lim.body_speed_axis_ms is None
                               else round(lim.body_speed_axis_ms, 6)),
        "body_speed_diagonal_ms": (None if lim.body_speed_diagonal_ms is None
                                   else round(lim.body_speed_diagonal_ms, 6)),
        "gear_ratio": lim.gear_ratio,
        "drive_joints": list(ids),
        "motor_side": {
            "stall_torque_nm": round(lim.motor_stall_torque_nm, 6),
            "no_load_rad_s": round(lim.motor_no_load_rad_s, 6),
            "torque_constant_nm_a": round(lim.torque_constant_nm_a, 6),
            "resistance_ohm": round(lim.resistance_ohm, 6),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(header) + yaml.safe_dump(body, sort_keys=False),
        encoding="utf-8")
    if write_sdf:
        sdf = output_path.with_suffix(".sdf")
        sdf.write_text("<!-- " + f"generated {stamp} from {plant_config};"
                       f" limits are at the WHEEL ({lim.gear_ratio:g}:1) -->\n"
                       + sdf_snippet(lim, list(ids)) + "\n", encoding="utf-8")
    return lim


def export_sweep_candidate(repo: Path, output_path: Path, *,
                           gear_ratio: float, wheel_radius_m: float,
                           total_mass_kg: float | None = None) -> WheelLimits:
    """Export a point from the selection sweep, without writing a plant file.

    The sweep searches ``(gear ratio, wheel radius, mass)``; picking a winner
    and handing it to Gazebo should not require regenerating
    ``config/generated/`` first and remembering to pass the matching ratio.
    This derives the motor for that drivetrain in memory -- the same code path
    ``sweep.drivetrain.build_variant`` uses -- so the exported limits belong to
    the candidate that was actually evaluated.
    """
    from ..sweep.drivetrain import build_variant

    cfg = build_variant(str(repo), gear_ratio, wheel_radius_m,
                        total_mass_kg if total_mass_kg else 15.0)
    plant = Path(output_path).with_name(Path(output_path).stem + "_plant.yaml")
    plant.parent.mkdir(parents=True, exist_ok=True)
    plant.write_text(
        f"# GENERATED for a sweep candidate: gear {gear_ratio:g}:1, "
        f"wheel {wheel_radius_m * 1000:.0f} mm"
        + (f", {total_mass_kg:g} kg" if total_mass_kg else "") + "\n"
        + yaml.safe_dump({
            "n_motors": 4,
            "torque_max_nm": float(cfg.motor.torque_max_nm),
            "omega_max_rad_s": float(cfg.motor.omega_max_rad_s),
            "torque_constant_nm_a": float(cfg.motor.torque_constant_nm_a),
            "resistance_ohm": float(cfg.motor.resistance_ohm),
            "inertia_kgm2": float(cfg.motor.inertia_kgm2),
        }, sort_keys=False), encoding="utf-8")
    # the body speeds come from the candidate's own jacobian. Leaving them
    # out here would omit the one number a candidate is chosen on, in the very
    # path built for choosing candidates.
    import numpy as np

    J = np.asarray(cfg.jacobian.drive_matrix, dtype=float)

    def body_speed(direction):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d[:2])
        return float(cfg.motor.omega_max_rad_s
                     / max(float(np.abs(J @ d).max()), 1e-9))

    return export_actuator_config(
        plant, output_path, gear_ratio=gear_ratio,
        wheel_radius_m=wheel_radius_m,
        joint_names=[w for w in ("fl", "fr", "rl", "rr")],
        body_speeds=(body_speed([1, 0, 0]), body_speed([1, 1, 0])))


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plant_config", type=Path)
    ap.add_argument("output_path", type=Path)
    ap.add_argument("--chassis", type=Path, default=None,
                    help="config/robot/chassis.yaml (for the gear ratio)")
    ap.add_argument("--gear-ratio", type=float, default=None)
    ap.add_argument("--wheel-radius", type=float, default=None)
    ap.add_argument("--candidate", metavar="GEAR,RADIUS[,MASS]", default=None,
                    help="export a sweep candidate: derives the motor for that "
                         "drivetrain instead of reading a plant file "
                         "(plant_config is then the repo root)")
    a = ap.parse_args(argv)
    try:
        if a.candidate:
            vals = [float(v) for v in a.candidate.split(",")]
            lim = export_sweep_candidate(
                a.plant_config, a.output_path, gear_ratio=vals[0],
                wheel_radius_m=vals[1],
                total_mass_kg=vals[2] if len(vals) > 2 else None)
            print(f"wrote {a.output_path} for gear {vals[0]:g}:1, "
                  f"wheel {vals[1] * 1000:.0f} mm\n"
                  f"  joint velocity {lim.velocity_limit_rad_s:.4f} rad/s\n"
                  f"  joint effort   {lim.effort_limit_nm:.4f} N m")
            return 0
        lim = export_actuator_config(a.plant_config, a.output_path,
                                     chassis_yaml=a.chassis,
                                     gear_ratio=a.gear_ratio,
                                     wheel_radius_m=a.wheel_radius)
    except ExportError as exc:
        print(f"export refused: {exc}")
        return 2
    body = ("" if lim.body_speed_diagonal_ms is None else
            f"  body speed    {lim.body_speed_axis_ms:.3f} m/s on axis, "
            f"{lim.body_speed_diagonal_ms:.3f} m/s diagonal\n")
    print(f"wrote {a.output_path}\n"
          f"  joint velocity {lim.velocity_limit_rad_s:.4f} rad/s "
          f"({lim.wheel_surface_speed_ms:.3f} m/s at the contact patch)\n"
          + body +
          f"  joint effort   {lim.effort_limit_nm:.4f} N m\n"
          f"  (motor side: {lim.motor_no_load_rad_s:.2f} rad/s, "
          f"{lim.motor_stall_torque_nm:.2f} N m at {lim.gear_ratio:g}:1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
