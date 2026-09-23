#!/usr/bin/env python3
"""Known-good (input, output) pairs from the real omni_sim_core.

The browser page runs a *copy* of ``mechanism/derive`` and
``mechanism/jacobian`` under Pyodide. That it loads and produces plausible
numbers proves nothing -- a page can be confidently wrong, and the failure
would look exactly like success. These pairs are the check: the same inputs go
in on both sides and the outputs have to match.

Run in the repo's own environment, which is where the pytest suite passes::

    .venv/bin/python scripts/export_golden_values.py

Writes ``omni_sim_web/tests/golden_values.json`` and, because a file:// page
cannot fetch its own siblings, the same content as
``omni_sim_web/js/golden.js``.

The cases are chosen to move every input the page exposes and to cross the
places the derivation branches: an unknown rotor inertia (estimated rather
than read), a datasheet override, and both ends of the gear-ratio range.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

import yaml  # noqa: E402

from omni_sim_core.mechanism.derive import derive_motor  # noqa: E402
from omni_sim_core.mechanism.jacobian import build_drive_jacobian  # noqa: E402
from omni_sim_core.field.slope import max_climb_angle_rad  # noqa: E402
from omni_sim_core.mechanism.schema import (Chassis, Datasheet,  # noqa: E402
                                            Physical)

PRESETS = ROOT / "config" / "presets" / "motors"
CHASSIS_YAML = ROOT / "config" / "robot" / "chassis.yaml"
OUT_JSON = ROOT / "omni_sim_web" / "tests" / "golden_values.json"
OUT_JS = ROOT / "omni_sim_web" / "js" / "golden.js"

# Relative tolerance for the browser comparison. Pyodide's numpy and CPython's
# both use IEEE 754 doubles and the arithmetic here is a handful of multiplies
# and one pseudo-inverse, so anything beyond rounding in the last couple of
# bits means the two sides are running different code, not different hardware.
TOLERANCE = 1e-9


def chassis_doc(base: dict, gear_ratio: float, wheel_radius: float,
                mass: float) -> dict:
    """The repo's own wheel layout with the page's three numbers applied.

    Must stay identical to chassis_doc() in omni_sim_web/js/web-api.js. Using
    the real config rather than an invented layout is deliberate: it is what
    surfaced that the shipped chassis.yaml has its wheel positions rotated
    90 deg against its declared drive axes.
    """
    doc = copy.deepcopy(base)
    doc["center_of_mass"]["mass_kg"] = mass
    for w in doc["drive_wheels"]:
        w["gear_ratio"] = gear_ratio
        w["radius_m"] = wheel_radius
    doc.pop("odometry", None)
    return doc


CASES = [
    # (name, preset, gear_ratio, wheel_radius, mass, datasheet overrides, why)
    ("nominal", "m3508_c620", 2.0, 0.050, 15.0, {},
     "mid-range, the drivetrain sweep's feasible region"),
    ("high_reduction", "m3508_c620", 6.0, 0.0508, 15.0, {},
     "the currently configured drivetrain"),
    ("direct_drive", "m3508_c620", 1.0, 0.075, 5.0, {},
     "both ends: least reduction, biggest wheel, lightest"),
    ("estimated_rotor_inertia", "generic_dc", 3.0, 0.038, 30.0, {},
     "rotor inertia is null in this preset, so derive estimates it"),
    ("datasheet_override", "m3508_c620", 2.5, 0.063, 22.5, {"kv_rpm_per_v": 28.0},
     "an override that re-derives Kt rather than being applied afterwards"),
]


def page_metrics(J, derived, mass_kg: float) -> dict:
    """The three figures the page shows, from golden-checked inputs.

    Included here so the browser's copy of these formulas is checked too.
    They are short, but they are the numbers a reader acts on, and a page
    whose headline figure is unverified is a page whose verification misses
    the point.

    No-load speed and stall push only -- no drag, no margin. Those belong to
    ``mechanism.robot_config``, which cannot be copied to the browser (it
    imports the simulator), and leaving them out keeps the page's numbers
    traceable to one place instead of two.
    """
    import numpy as np

    def body_speed(direction):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d[:2])
        return float(derived.no_load_speed_rad_s
                     / max(float(np.abs(J @ d).max()), 1e-9))

    # stall push along +x: wheel torque -> body wrench is J^T tau, and the
    # wheels must turn the way the motion needs
    tau = np.sign(J @ np.array([1.0, 0.0, 0.0])) * derived.stall_torque_nm
    push_x = float((J.T @ tau)[0])

    yaw = np.abs(J[:, 2])
    return {
        "v_noload_axis_ms": body_speed([1.0, 0.0, 0.0]),
        "v_noload_diag_ms": body_speed([1.0, 1.0, 0.0]),
        "push_stall_n": push_x,
        "climb_deg_at_stall": float(np.degrees(max_climb_angle_rad(push_x, mass_kg))),
        # Per-wheel share of the yaw column. A well-formed X layout gives all
        # four the same value; the shipped chassis.yaml gives two of them 0.
        "yaw_authority": [float(v) for v in yaw / max(float(yaw.max()), 1e-12)],
    }


def _derived_to_dict(d) -> dict:
    return {
        "torque_constant_nm_a": d.torque_constant_nm_a,
        "back_emf_v_s": d.back_emf_v_s,
        "resistance_ohm": d.resistance_ohm,
        "stall_torque_nm": d.stall_torque_nm,
        "no_load_speed_rad_s": d.no_load_speed_rad_s,
        "rotor_inertia_kgm2": d.rotor_inertia_kgm2,
        "reflected_inertia_kgm2": d.reflected_inertia_kgm2,
        "load_contribution_kgm2": d.load_contribution_kgm2,
        "damping_nms": d.damping_nms,
        "estimated": sorted(d.estimated),
    }


def build() -> dict:
    presets = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
               for p in sorted(PRESETS.glob("*.yaml"))}
    base_chassis = yaml.safe_load(CHASSIS_YAML.read_text(encoding="utf-8"))
    cases = []
    for name, preset, gear, radius, mass, overrides, why in CASES:
        doc = presets[preset]
        ds_doc = dict(doc["datasheet"])
        ds_doc.update(overrides)
        datasheet = Datasheet.from_doc(ds_doc, f"preset '{preset}'")
        physical = Physical.from_doc(doc.get("physical"))

        chassis = Chassis.from_doc(
            chassis_doc(base_chassis, gear, radius, mass))
        wheel = chassis.drive_wheels[0]
        # same load model as mechanism/build.derive_reference_motor
        load = (mass / len(chassis.drive_wheels)) * wheel.radius_m ** 2
        derived = derive_motor(datasheet, physical, gear_ratio=gear,
                               load_inertia_wheel_side_kgm2=load)
        J = build_drive_jacobian(chassis)
        cases.append({
            "name": name,
            "why": why,
            "input": {"preset": preset, "gear_ratio": gear,
                      "wheel_radius_m": radius, "total_mass_kg": mass,
                      "datasheet_overrides": overrides},
            "expected": {
                "derived": _derived_to_dict(derived),
                "drive_matrix": [[float(v) for v in row] for row in J],
                "page": page_metrics(J, derived, mass),
            },
        })
    return {"tolerance": TOLERANCE,
            "source": "scripts/export_golden_values.py against omni_sim_core/src",
            "cases": cases}


def main() -> int:
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text(
        "// GENERATED by scripts/export_golden_values.py -- do not edit.\n"
        "// Same content as tests/golden_values.json; a file:// page cannot\n"
        "// fetch the .json, so it also ships as a script.\n"
        "window.OMNI_SIM_GOLDEN = " + json.dumps(data) + ";\n",
        encoding="utf-8")
    print(f"{len(data['cases'])} cases, tolerance {data['tolerance']:g}")
    for c in data["cases"]:
        d = c["expected"]["derived"]
        pg = c["expected"]["page"]
        print(f"  {c['name']:24s} Kt {d['torque_constant_nm_a']:.6f}  "
              f"tau {d['stall_torque_nm']:7.4f}  "
              f"v_axis {pg['v_noload_axis_ms']:5.2f} m/s  "
              f"climb {pg['climb_deg_at_stall']:4.1f} deg  "
              f"yaw {[round(v, 2) for v in pg['yaw_authority']]}")
    print(f"-> {OUT_JSON.relative_to(ROOT)}\n-> {OUT_JS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
