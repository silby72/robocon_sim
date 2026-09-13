"""P2 field-layer acceptance + unit tests (依頼 §3).

Numbered ``acceptance N:`` comments map to the 6 P2 criteria. Headless, no
rclpy/PySide6, deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from omni_sim_core.env.occupancy_grid import OCCUPIED, OccupancyGrid
from omni_sim_core.field import FieldSpec, LayeredField
from omni_sim_core.field.primitives import FREE, OCC, UNKNOWN

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "config" / "field" / "robocon2027.yaml"


@pytest.fixture
def field():
    return LayeredField.from_yaml(SPEC)


# --------------------------------------------------------------------------- #
def test_acceptance1_deterministic_bit_identical():
    """acceptance 1: two builds from the same spec give bit-identical rasters."""
    f1 = LayeredField.from_yaml(SPEC)
    f2 = LayeredField.from_yaml(SPEC)
    for layer in ("ground", "l1", "l2"):
        for purpose in ("nav", "loc"):
            assert np.array_equal(f1.raster(layer, purpose),
                                  f2.raster(layer, purpose))


def test_acceptance2_localization_interior_is_unknown(field):
    """acceptance 2: inside an obstacle the localization grid is UNKNOWN."""
    loc = field.raster("l1", "loc")
    g = field.grid("l1", "loc")
    # centre of the L2 slab (an obstacle seen from L1), away from its outline
    r, c = g.world_to_grid(5.5, 5.5)
    assert loc[r, c] == UNKNOWN
    # and somewhere on that outline is occupied (surface is drawn)
    assert (loc == OCC).any()


def test_acceptance3_nav_interior_is_occupied(field):
    """acceptance 3: inside an obstacle the nav grid is OCCUPIED (filled)."""
    nav = field.raster("l1", "nav")
    g = field.grid("l1", "nav")
    r, c = g.world_to_grid(5.5, 5.5)      # L2 slab centre
    assert nav[r, c] == OCC


def test_acceptance4_barrier_height_vs_lidar_plane():
    """acceptance 4: raising l1_barrier_h above the LiDAR plane makes the L1
    perimeter appear in the localization grid; below it, it vanishes."""
    # L1 LiDAR plane = 600 + 185 = 785 mm. Barrier top = 600 + l1_barrier_h.
    tall = LayeredField.from_yaml(SPEC, unknowns={"l1_barrier_h": 300})   # top 900 > 785
    short = LayeredField.from_yaml(SPEC, unknowns={"l1_barrier_h": 100})  # top 700 < 785
    g = tall.grid("l1", "loc")
    r, c = g.world_to_grid(2.5, 5.5)      # a point on the west perimeter barrier
    assert tall.raster("l1", "loc")[r, c] == OCC
    assert short.raster("l1", "loc")[r, c] != OCC


def test_acceptance5_pgm_yaml_roundtrips(field, tmp_path):
    """acceptance 5: generated PGM+YAML is read back by occupancy_grid.py."""
    paths = field.write(tmp_path, layers=["ground"], purposes=("nav",))
    yaml_path = paths[0]
    og = OccupancyGrid.from_yaml(yaml_path)
    nav = field.raster("ground", "nav")
    # occupied cells must survive the write/flip/read round-trip
    expect = (nav == OCC)
    assert og.grid.shape == nav.shape
    assert np.array_equal(og.grid == OCCUPIED, expect)


def test_acceptance6_thin_wall_is_one_cell(field):
    """acceptance 6: a sub-cell divider (30 mm at 20 mm/cell) renders >=1 cell."""
    nav = field.raster("ground", "nav")
    g = field.grid("ground", "nav")
    # scan across x at several y along the southern divider (x=5.5, y in [0,1])
    for y in (0.1, 0.5, 0.9):
        r, _ = g.world_to_grid(5.5, y)
        row = nav[r]
        occ_cols = np.where(row == OCC)[0]
        assert occ_cols.size >= 1, f"divider vanished at y={y}"
        # thin: the divider itself spans no more than 2 cells here
        # (boundary/other obstacles are elsewhere on the row)
        c_mid = g.world_to_grid(5.5, y)[1]
        near = occ_cols[np.abs(occ_cols - c_mid) <= 2]
        assert 1 <= near.size <= 3


def test_thin_wall_enforced_min_thickness():
    """A wall thinner than a cell is floored to one cell at build time."""
    import tempfile
    spec_text = """
units: mm
resolution: 100
field: {size: [1000, 1000], origin: [0, 0]}
layers: {ground: {floor_z: 0}}
lidar_height: 100
robot_band: [0, 300]
geometry:
  - {name: wall, type: segment, seg: [500, 0, 500, 1000], thickness: 30, z: [0, 300]}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(spec_text)
        path = fh.name
    field = LayeredField.from_yaml(path)
    nav = field.raster("ground", "nav")
    # 30 mm wall at 100 mm/cell -> enforced to 1 cell, present on every row
    assert np.all((nav == OCC).any(axis=1))


def test_geometry_returns_layer_obstacles(field):
    """geometry(layer) returns the exact primitives that can collide there."""
    names_l1 = {p.name for p in field.geometry("l1")}
    assert "l2_platform" in names_l1          # L2 slab is a wall on L1
    assert "mustika_pillar" not in names_l1   # it only reaches 500 mm, below L1


def test_l1_hole_is_lethal_in_nav(field):
    """The Mustika pocket punched through L1 is a drop -> lethal in nav."""
    nav = field.raster("l1", "nav")
    g = field.grid("l1", "nav")
    r, c = g.world_to_grid(5.5, 8.0)          # inside the 1x1 m pocket
    assert nav[r, c] == OCC


def test_mm_to_si_conversion(field):
    assert field.spec.size == (11.0, 11.0)
    assert field.spec.resolution == 0.02
    assert field.spec.layers["l1"].floor_z == 0.6
