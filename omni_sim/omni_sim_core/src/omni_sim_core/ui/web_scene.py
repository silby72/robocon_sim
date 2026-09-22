"""One JSON scene description for every non-matplotlib front-end.

The browser dashboard used to carry its own hand-copied subset of the field
(a dozen coloured rectangles) and drew the robot as a fixed 9 px dot. Both
were lies of different kinds: the dot has no size, so nothing on screen could
show *why* a 0.9 m body is blocked by a 0.23 m gap, and the copied rectangles
drifted from ``field_layout_2027`` / the field spec the moment either changed.

So this module exports what the simulator actually judges against:

    zones      -- ``field_layout_2027._zones()`` verbatim (the painted floor)
    walls      -- the real ``FieldSpec`` primitives, each tagged with the
                  layers whose robot band it protrudes into
    layers     -- floor height, drivable extent and holes (off the extent is
                  a fall, which is every bit as lethal as a wall)
    connectors -- the ramps, which carve *through* a wall on both layers
    robot      -- the footprint polygon and wheels from ``chassis.yaml``

``walls[i].blocks`` is the part worth being careful about: it mirrors
``slicer.slice_nav``'s band test exactly (``overlaps_band(floor + band_lo,
floor + band_hi)``), so a front-end that dims the walls not in ``blocks`` for
the robot's current level is showing the same "wall / not a wall" decision the
collision checker made, not a plausible-looking approximation of it.

Everything is in millimetres, because every consumer of this is a drawing
surface and the rulebook is in millimetres; only ``r_circ_mm`` is derived.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import field_layout_2027 as fl


def _hex(rgb) -> str | None:
    if rgb is None:
        return None
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def zone_scene() -> list[dict[str, Any]]:
    """The painted floor: every static zone the matplotlib viewer draws.

    Same call the viewer makes, so the two front-ends cannot disagree about
    where the Start Zone or the Stairs are.
    """
    out = []
    for kind, geom, color, zorder, label in fl._zones():
        out.append({"kind": kind, "geom": [float(v) for v in geom],
                    "color": _hex(color), "z": float(zorder), "label": label})
    return out


def wall_scene(field_yaml: str | Path) -> dict[str, Any]:
    """Walls, layers and connectors straight out of the field spec.

    Wall geometry comes from ``Primitive.shape_2d()`` -- the same single source
    of truth the rasteriser and the distance checks use (see
    ``field/primitives.py``), so what is drawn is what is hit.
    """
    from ..field.spec import FieldSpec

    spec = FieldSpec.from_yaml(field_yaml)
    prims = spec.build_primitives()
    band_lo, band_hi = spec.robot_band

    walls: list[dict[str, Any]] = []
    for p in prims:
        kind, *rest = p.shape_2d()
        entry: dict[str, Any] = {
            "name": p.name,
            "z": [p.z_lo * 1000.0, p.z_hi * 1000.0],
            # Which layers is this a wall *on*? Exactly slice_nav's test: a
            # primitive is an obstacle for a layer only when it pokes into that
            # layer's robot band. The L1 slab is a 600 mm wall from the ground
            # and invisible from on top of it -- that asymmetry is this list.
            "blocks": [name for name, layer in spec.layers.items()
                       if p.overlaps_band(layer.floor_z + band_lo,
                                          layer.floor_z + band_hi)],
        }
        if kind == "circle":
            center, r = rest
            entry.update(kind="circle", center=[float(center[0]) * 1000.0,
                                                float(center[1]) * 1000.0],
                         r=float(r) * 1000.0)
        else:
            entry.update(kind="polygon",
                         points=[[float(x) * 1000.0, float(y) * 1000.0]
                                 for x, y in rest[0]])
        walls.append(entry)

    layers = {
        name: {
            "floor_z": layer.floor_z * 1000.0,
            # None means "the whole field is drivable" (the ground layer);
            # otherwise everything outside this rectangle is a fall.
            "extent": (None if layer.extent is None
                       else [v * 1000.0 for v in layer.extent]),
            "holes": [[v * 1000.0 for v in h] for h in layer.holes],
        }
        for name, layer in spec.layers.items()
    }
    # Connectors carry their slope so a viewer can show which way is uphill
    # and how steep: a ramp drawn as a flat rectangle is indistinguishable
    # from a painted stripe.
    from ..field.slope import SlopeField

    slopes = {s.name: s for s in SlopeField.from_spec(spec).slopes}
    connectors = []
    for c in spec.connectors:
        entry = {"name": c.name, "rect": [v * 1000.0 for v in c.rect],
                 "links": list(c.links)}
        s = slopes.get(c.name)
        if s is not None:
            entry.update(uphill=list(s.uphill()), pitch=round(s.angle_deg, 2),
                         rise=s.rise * 1000.0, run=s.run * 1000.0)
        connectors.append(entry)
    return {"walls": walls, "layers": layers, "connectors": connectors,
            "size": [spec.size[0] * 1000.0, spec.size[1] * 1000.0],
            "resolution": spec.resolution * 1000.0,
            "robot_band": [band_lo * 1000.0, band_hi * 1000.0]}


def robot_scene(chassis_yaml: str | Path) -> dict[str, Any]:
    """The configured chassis: what ``omni_sim_gui``'s chassis page draws.

    Body frame, millimetres, REP-103 (+x forward, +y left). The wheels carry
    their drive axis so a front-end can draw the X-configuration the right way
    round rather than four identical blobs -- getting that wrong looks fine
    until someone tries to read a wheel-speed overlay off it.
    """
    from ..mechanism.yaml_rt import rt_load
    from ..mechanism.schema import Chassis

    chassis = Chassis.from_doc(rt_load(str(chassis_yaml)))
    size_mm = chassis.footprint.size_m * 1000.0
    half = size_mm / 2.0
    return {
        "shape": chassis.footprint.shape,
        "size": size_mm,
        # closed polygon, so a canvas can stroke it without repeating the first
        # point itself; also the thing collision judges (rotated, not a circle)
        "footprint": [[half, half], [-half, half], [-half, -half], [half, -half]],
        # planning inflates by this; the gap between it and half the side
        # length is the entire spare clearance budget (evaluation/footprint.py)
        "r_circ": 0.5 * (size_mm ** 2 + size_mm ** 2) ** 0.5,
        "r_insc": half,
        "com": [chassis.center_of_mass.position_m[0] * 1000.0,
                chassis.center_of_mass.position_m[1] * 1000.0],
        "mass_kg": chassis.center_of_mass.mass_kg,
        "wheels": [{"id": w.id,
                    "pos": [w.position_m[0] * 1000.0, w.position_m[1] * 1000.0],
                    "axis": w.drive_axis_rad,
                    "radius": w.radius_m * 1000.0,
                    "reverse": bool(w.reverse)}
                   for w in chassis.drive_wheels],
        "odometry": [{"id": o.id, "type": o.type,
                      "pos": [o.position_m[0] * 1000.0, o.position_m[1] * 1000.0],
                      "axis": o.measure_axis_rad}
                     for o in chassis.odometry],
    }


def build_scene(field_yaml: str | Path | None = None,
                chassis_yaml: str | Path | None = None,
                team: str = "red",
                motors: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the full scene. Both files are optional: a missing chassis
    just means a front-end has no body to draw, which is exactly the state
    this module exists to get out of, so it is reported rather than hidden."""
    scene: dict[str, Any] = {
        "team": team,
        "field": {"size": fl.FIELD_SIZE_MM, "center_x": fl.CENTER_X_MM},
        "zones": zone_scene(),
        # the only places a level change is legal (ui/leveled_field_2027.py)
        "gates": {
            "ramp": list(fl.RAMP_GATE_RECT_MM if team == "red"
                         else fl._mirror(fl.RAMP_GATE_RECT_MM)),
            "stairs": list(fl.STAIRS_GATE_MARGIN_MM),
        },
        "missing": [],
    }
    # {"available": [...], "current": "...", "spec": {...}} when the caller can
    # actually swap the drivetrain; a front-end without this renders no picker
    # rather than one that silently does nothing.
    if motors is not None:
        scene["motors"] = motors
    if field_yaml and Path(field_yaml).exists():
        scene.update(wall_scene(field_yaml))
    else:
        scene["missing"].append("field")
    if chassis_yaml and Path(chassis_yaml).exists():
        scene["robot"] = robot_scene(chassis_yaml)
    else:
        scene["missing"].append("chassis")
    return scene


def scene_json(**kwargs) -> str:
    return json.dumps(build_scene(**kwargs), separators=(",", ":"))
