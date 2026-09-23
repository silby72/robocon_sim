// Hand-written glue -- NOT generated, NOT a copy of omni_sim_core.
//
// Everything in core-bundle.js is a byte-for-byte copy of the real modules.
// This file is the small amount of Python that does not exist in the repo:
// turning the page's four sliders into the documents mechanism/schema wants,
// and turning the results back into something JSON-serialisable.
//
// It lives in a .js file rather than a .py file for one reason: the no-server
// rule. A file:// page cannot fetch() a sibling .py, and there is no build
// step to inline one, so the source rides in as a raw string literal.
//
// The formulas at the bottom (top speed, stall push, climb angle) are the only
// arithmetic here that duplicates logic living elsewhere in the repo -- the
// modules that own them, mechanism/robot_config and field/slope, both import
// the simulator or the field spec and cannot be copied to the browser. They
// are duplicated under protest and under test: scripts/export_golden_values.py
// computes the same four numbers through the *real* max_climb_angle_rad, and
// tests/verify.html fails if this copy disagrees at 1e-9.

window.OMNI_WEB_API_PY = String.raw`
"""Glue between the chassis page's controls and omni_sim_core."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from omni_sim_core.mechanism.derive import derive_motor
from omni_sim_core.mechanism.jacobian import build_drive_jacobian
from omni_sim_core.mechanism.schema import Chassis, Datasheet, Physical

G = 9.80665          # mirrors omni_sim_core.field.slope.G


def chassis_doc(base, gear_ratio, wheel_radius, mass):
    """The repo's own wheel layout, with the page's three numbers applied.

    The positions and drive axes come from config/robot/chassis.yaml -- the
    machine actually being built -- rather than an invented layout. Only what
    the sliders control is overridden.

    No backticks anywhere below this line: the whole module is carried in a
    JS String.raw template literal, and a backtick ends it. reST markup in
    these docstrings breaks the page and nothing in pytest can see it.
    """
    doc = copy.deepcopy(base)
    doc["center_of_mass"]["mass_kg"] = mass
    for w in doc["drive_wheels"]:
        w["gear_ratio"] = gear_ratio
        w["radius_m"] = wheel_radius
    doc.pop("odometry", None)          # not this page's subject
    return doc


def page_metrics(J, derived, mass_kg):
    """The four figures the page leads with.

    Kept identical to page_metrics() in scripts/export_golden_values.py; the
    verification page compares them at 1e-9, so a change to one that is not
    made to the other turns red rather than drifting quietly.
    """
    def body_speed(direction):
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d[:2])
        return float(derived.no_load_speed_rad_s
                     / max(float(np.abs(J @ d).max()), 1e-9))

    tau = np.sign(J @ np.array([1.0, 0.0, 0.0])) * derived.stall_torque_nm
    push_x = float((J.T @ tau)[0])

    yaw = np.abs(J[:, 2])
    yaw_share = [float(v) for v in (yaw / max(float(yaw.max()), 1e-12))]

    ratio = 2.0 * push_x / (mass_kg * G)
    climb = math.pi / 4 if ratio >= 1.0 else 0.5 * math.asin(ratio)
    return {
        "v_noload_axis_ms": body_speed([1.0, 0.0, 0.0]),
        "v_noload_diag_ms": body_speed([1.0, 1.0, 0.0]),
        "push_stall_n": push_x,
        "climb_deg_at_stall": math.degrees(climb),
        # Per-wheel yaw authority, normalised. A well-formed X layout gives
        # all four the same share. The shipped chassis.yaml does not: its
        # wheel positions are rotated 90 deg against its declared drive axes,
        # so two wheels come out at 0.000 and yaw is produced by the other two
        # in opposition. Reported rather than silently averaged away, because
        # a number nobody computes is a defect nobody finds.
        "yaw_authority": yaw_share,
    }


def evaluate(preset_doc, base_chassis, gear_ratio, wheel_radius, mass,
             overrides=None):
    """One chassis, start to finish. Returns a plain dict, ready for JSON."""
    ds_doc = dict(preset_doc["datasheet"])
    if overrides:
        ds_doc.update({k: v for k, v in overrides.items() if v is not None})
    datasheet = Datasheet.from_doc(ds_doc, "browser input")
    physical = Physical.from_doc(preset_doc.get("physical"))

    chassis = Chassis.from_doc(
        chassis_doc(base_chassis, gear_ratio, wheel_radius, mass))
    wheel = chassis.drive_wheels[0]
    # same load model as mechanism/build.derive_reference_motor
    load = (mass / len(chassis.drive_wheels)) * wheel.radius_m ** 2

    derived = derive_motor(datasheet, physical, gear_ratio=gear_ratio,
                           load_inertia_wheel_side_kgm2=load)
    J = build_drive_jacobian(chassis)
    return {
        "derived": {
            "torque_constant_nm_a": derived.torque_constant_nm_a,
            "back_emf_v_s": derived.back_emf_v_s,
            "resistance_ohm": derived.resistance_ohm,
            "stall_torque_nm": derived.stall_torque_nm,
            "no_load_speed_rad_s": derived.no_load_speed_rad_s,
            "rotor_inertia_kgm2": derived.rotor_inertia_kgm2,
            "reflected_inertia_kgm2": derived.reflected_inertia_kgm2,
            "load_contribution_kgm2": derived.load_contribution_kgm2,
            "damping_nms": derived.damping_nms,
            "estimated": sorted(derived.estimated),
        },
        "drive_matrix": [[float(v) for v in row] for row in J],
        "page": page_metrics(J, derived, mass),
        # What the datasheet block *declares*, beside what the model actually
        # used. derive_motor never reads stall_torque_nm or no_load_speed_rpm
        # -- it recomputes both -- so a person transcribing a datasheet can
        # type a stall torque that changes nothing. Showing the pair is the
        # only way that is visible without reading derive.py.
        "declared": {
            "stall_torque_nm": float(ds_doc["stall_torque_nm"]),
            "no_load_speed_rpm": float(ds_doc["no_load_speed_rpm"]),
            "torque_ratio": (derived.stall_torque_nm
                             / float(ds_doc["stall_torque_nm"])),
            "speed_ratio": (derived.no_load_speed_rad_s * 60.0 / (2.0 * math.pi)
                            / float(ds_doc["no_load_speed_rpm"])),
        },
        "geometry": {
            "footprint_m": chassis.footprint.size_m,
            "wheels": [
                {"id": w.id, "x": w.position_m[0], "y": w.position_m[1],
                 "axis_rad": w.drive_axis_rad, "radius_m": w.radius_m}
                for w in chassis.drive_wheels
            ],
        },
    }


def evaluate_json(payload_json):
    """String in, string out. Keeps the JS side free of proxy lifetimes."""
    q = json.loads(payload_json)
    try:
        out = evaluate(q["preset"], q["chassis"], q["gear_ratio"],
                       q["wheel_radius_m"], q["total_mass_kg"],
                       q.get("datasheet_overrides"))
        return json.dumps({"ok": True, "result": out})
    except Exception as exc:                       # a bad slider combination
        return json.dumps({"ok": False,
                           "error": "%s: %s" % (type(exc).__name__, exc)})


def smoke():
    """Step 3 of the brief: prove numpy crossed the bridge, nothing more."""
    return json.dumps({
        "numpy": np.__version__,
        "array": (np.arange(4.0) * 0.5).tolist(),
        "pinv_identity": bool(np.allclose(
            np.linalg.pinv(np.eye(3)), np.eye(3))),
    })
`;
