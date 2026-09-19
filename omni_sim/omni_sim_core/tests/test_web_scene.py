"""The browser scene must describe the field the *simulator* judges against.

The whole point of ``ui/web_scene`` is that a front-end stops carrying its own
copy of the field. These tests fail if that copy creeps back in by drift: they
re-derive the nav rasters from the spec and check the exported walls agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.field.primitives import OCC
from omni_sim_core.field.slicer import slice_nav
from omni_sim_core.field.spec import FieldSpec
from omni_sim_core.ui.web_scene import build_scene

ROOT = Path(__file__).resolve().parents[2]
FIELD_YAML = ROOT / "config" / "field" / "robocon2027.yaml"
CHASSIS_YAML = ROOT / "config" / "robot" / "chassis.yaml"

pytestmark = pytest.mark.skipif(not FIELD_YAML.exists(),
                                reason="repo config/ not available")


@pytest.fixture(scope="module")
def scene():
    return build_scene(FIELD_YAML, CHASSIS_YAML)


def _representative_point(wall) -> tuple[float, float]:
    """A point in metres that is inside the wall's own shape."""
    if wall["kind"] == "circle":
        return (wall["center"][0] / 1000.0, wall["center"][1] / 1000.0)
    pts = np.array(wall["points"], dtype=float) / 1000.0
    return tuple(pts.mean(axis=0))


def test_scene_is_json_serialisable(scene):
    # numpy scalars would sail through the dataclasses and blow up only at the
    # publisher, where the traceback points at rclpy instead of at this module
    json.dumps(scene)
    assert scene["missing"] == []


def test_robot_comes_from_the_chassis_config(scene):
    robot = scene["robot"]
    assert robot["size"] == pytest.approx(900.0)          # chassis.yaml, mm
    assert len(robot["footprint"]) == 4                   # a rectangle, not a dot
    assert robot["r_circ"] == pytest.approx(900.0 * np.sqrt(2) / 2)
    assert {w["id"] for w in robot["wheels"]} == {"fl", "fr", "rl", "rr"}
    for w in robot["wheels"]:
        assert w["radius"] > 0


def test_every_wall_blocks_exactly_the_layers_its_raster_does(scene):
    """``walls[].blocks`` is what a front-end greys out; it has to be the same
    decision slice_nav made, not a plausible-looking approximation."""
    spec = FieldSpec.from_yaml(FIELD_YAML)
    prims = spec.build_primitives()
    res = spec.resolution
    ox, oy = spec.origin
    rasters = {name: slice_nav(spec, prims, layer)
               for name, layer in spec.layers.items()}

    def carved(x: float, y: float, level: str) -> bool:
        return any(level in c.links
                   and c.rect[0] <= x <= c.rect[2] and c.rect[1] <= y <= c.rect[3]
                   for c in spec.connectors)

    for wall in scene["walls"]:
        x, y = _representative_point(wall)
        col, row = int((x - ox) / res), int((y - oy) / res)
        row = min(max(row, 0), rasters["ground"].shape[0] - 1)
        col = min(max(col, 0), rasters["ground"].shape[1] - 1)
        for level in wall["blocks"]:
            if carved(x, y, level):
                continue   # a ramp carves its doorway through the wall, last
            assert rasters[level][row, col] == OCC, (
                f"{wall['name']} claims to block {level} but the nav raster "
                f"is free at ({x:.2f}, {y:.2f})")


def test_the_l1_barrier_is_a_wall_on_l1_and_nothing_on_the_ground(scene):
    """The asymmetry the dot-and-sketch dashboard could never show: the same
    600 mm-high barrier is solid from above and simply absent from below."""
    barriers = [w for w in scene["walls"] if w["name"].startswith("l1_barrier")]
    assert barriers, "the L1 perimeter barrier vanished from the spec"
    for b in barriers:
        assert b["blocks"] == ["l1"]

    slab = next(w for w in scene["walls"] if w["name"] == "l1_platform")
    assert slab["blocks"] == ["ground"]   # a 600 mm wall from below only


def test_ramps_are_exported_so_a_viewer_can_show_the_doorway(scene):
    names = {c["name"] for c in scene["connectors"]}
    assert names == {"ramp_red", "ramp_blue"}
    for c in scene["connectors"]:
        assert set(c["links"]) == {"ground", "l1"}


def test_layer_extents_carry_the_fall_edges(scene):
    layers = scene["layers"]
    assert layers["ground"]["extent"] is None     # the whole field is drivable
    assert layers["l1"]["extent"] == [2500.0, 2500.0, 8500.0, 8500.0]
    assert layers["l1"]["holes"]                  # the Mustika pocket is a drop
