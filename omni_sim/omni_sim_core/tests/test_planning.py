"""P1 planning-layer acceptance + unit tests (依頼 §2, DECISIONS.md §2).

Numbered ``acceptance N:`` comments map to the 12 P1 acceptance criteria. The
whole file runs headless with no rclpy / PySide6 / GUI (invariants 1, 2) and is
deterministic (invariant 6).
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from omni_sim_core.env.occupancy_grid import (FREE, OCCUPIED, MapMetadata,
                                              OccupancyGrid, generate_rect_field)
from omni_sim_core.planning import (CostField, GridPlanner, PlanConfig, plan,
                                    build_orientation_profile, shortcut_path,
                                    line_of_sight)
from omni_sim_core.planning.corner import (corner_radii, corner_speed_limits,
                                           tangent_offsets)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def empty_grid(w=6.0, h=6.0, res=0.05):
    """Open field, no border walls -> a clean diagonal is available."""
    cols = int(round(w / res))
    rows = int(round(h / res))
    grid = np.full((rows, cols), FREE, dtype=np.uint8)
    return OccupancyGrid(grid, MapMetadata(resolution=res, origin=(0.0, 0.0, 0.0)))


def grid_with_block(w=6.0, h=6.0, res=0.05, block=(2.5, 2.5, 3.5, 3.5)):
    g = empty_grid(w, h, res)
    x0, y0, x1, y1 = block
    r0, c0 = g.world_to_grid(x0, y0)
    r1, c1 = g.world_to_grid(x1, y1)
    g.grid[r0:r1, c0:c1] = OCCUPIED
    return g


def hollow_block_grid(w=6.0, h=6.0, res=0.05):
    """A small square obstacle drawn as an OUTLINE only -- interior is free.

    This is the case the EDT sealing must handle (§2.4): the interior is a small
    ENCLOSED free pocket, so it gets sealed to occupied; a bordered arena's large
    interior would not.
    """
    g = empty_grid(w, h, res)
    r0, c0 = g.world_to_grid(2.5, 2.5)
    r1, c1 = g.world_to_grid(3.5, 3.5)  # 1.0 x 1.0 m ~ 2.8% of a 6x6 map
    g.grid[r0:r1, c0:c1] = OCCUPIED
    g.grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = FREE  # hollow it out
    return g


# --------------------------------------------------------------------------- #
# cost field
# --------------------------------------------------------------------------- #
def test_acceptance2_obstacle_interior_edt_is_zero():
    """acceptance 2: EDT distance inside a (hollow) obstacle is exactly 0."""
    g = hollow_block_grid()
    cf = CostField(g, PlanConfig(r_circ=0.3))
    r0, c0 = g.world_to_grid(2.5, 2.5)
    r1, c1 = g.world_to_grid(3.5, 3.5)
    interior = cf.dist[r0 + 2:r1 - 2, c0 + 2:c1 - 2]
    assert np.all(interior == 0.0), "binary_fill_holes must seal the interior"
    # and the interior must be lethal, not a reachable pocket
    assert np.all(cf.lethal[r0 + 2:r1 - 2, c0 + 2:c1 - 2])


def test_bordered_arena_interior_stays_free():
    """Regression: a walled arena's interior must NOT be sealed (only small
    enclosed pockets are). Naive binary_fill_holes would kill the whole field."""
    g = generate_rect_field(12.0, 12.0, 0.05)  # 240x240, 1-cell border wall
    cf = CostField(g, PlanConfig(r_circ=0.3))
    cr, cc = g.world_to_grid(6.0, 6.0)
    assert not cf.lethal[cr, cc]          # centre is drivable
    assert not cf.occupied[cr, cc]        # and was not sealed


def test_cost_field_zones():
    g = grid_with_block()
    cf = CostField(g, PlanConfig(r_circ=0.3, soft_weight=2.0, soft_k=4.0))
    # far from any wall -> zero soft cost
    fr, fc = g.world_to_grid(0.5, 0.5)
    assert cf.soft_cost[fr, fc] == 0.0
    assert not cf.lethal[fr, fc]
    # just outside r_circ -> positive soft cost
    assert cf.soft_cost.max() > 0.0


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
def test_acceptance1_empty_field_path_is_diagonal():
    """acceptance 1: obstacle-free, path length == straight diagonal (dx==dy)."""
    g = empty_grid(6.0, 6.0, 0.05)
    start, goal = (0.5, 0.5), (5.0, 5.0)  # dx == dy -> octile == euclidean
    res = g.meta.resolution
    r = plan(g, start, goal, PlanConfig(r_circ=res, soft_weight=0.0))
    assert r.success
    diag = np.hypot(goal[0] - start[0], goal[1] - start[1])
    # allow one cell of discretisation slack
    assert r.path.length == pytest.approx(diag, abs=2 * res)


def test_acceptance3_path_avoids_lethal():
    """acceptance 3: every path vertex is on a non-lethal cell."""
    g = grid_with_block()
    cfg = PlanConfig(r_circ=0.3)
    cf = CostField(g, cfg)
    r = GridPlanner(cf).plan((0.5, 0.5), (5.5, 5.5))
    assert r.success
    for x, y in r.path.points:
        assert not cf.is_lethal_world(x, y)


def test_acceptance4_higher_soft_weight_pushes_off_wall():
    """acceptance 4: raising w moves the path away from the wall."""
    g = grid_with_block()
    cf_lo = CostField(g, PlanConfig(r_circ=0.3, soft_weight=0.0))
    cf_hi = CostField(g, PlanConfig(r_circ=0.3, soft_weight=30.0, soft_k=2.0))
    r_lo = GridPlanner(cf_lo).plan((0.5, 0.5), (5.5, 5.5))
    r_hi = GridPlanner(cf_hi).plan((0.5, 0.5), (5.5, 5.5))
    assert r_lo.success and r_hi.success

    def min_clearance(cf, path):
        return min(cf.dist[g.world_to_grid(x, y)] for x, y in path.points)

    # the higher-weight path keeps more distance from the obstacle
    assert min_clearance(cf_hi, r_hi.path) > min_clearance(cf_lo, r_lo.path)


def test_acceptance5_no_path_returns_failed_not_exception():
    """acceptance 5: unreachable goal -> PlanResult.failed, no exception."""
    g = empty_grid(6.0, 6.0, 0.05)
    # wall right across the field, sealing the goal off
    r0, _ = g.world_to_grid(0, 3.0)
    r1, _ = g.world_to_grid(0, 3.2)
    g.grid[r0:r1, :] = OCCUPIED
    r = plan(g, (3.0, 1.0), (3.0, 5.0), PlanConfig(r_circ=0.2))
    assert r.success is False and r.path is None
    assert r.expanded_nodes > 0


def test_acceptance6_start_in_inflation_escapes_with_warning():
    """acceptance 6: start inside inflation -> shift out + warning recorded."""
    g = grid_with_block()
    # start point sits just inside the inflated region next to the block
    r = plan(g, (2.45, 2.45), (5.5, 5.5), PlanConfig(r_circ=0.3, max_shift=1.0))
    assert r.success
    assert any("start" in w for w in r.warnings)


def test_acceptance7_goal_in_inflation_escapes_with_warning():
    """acceptance 7: goal inside inflation -> shift out + warning recorded."""
    g = grid_with_block()
    r = plan(g, (0.5, 0.5), (3.55, 3.55), PlanConfig(r_circ=0.3, max_shift=1.0))
    assert r.success
    assert any("goal" in w for w in r.warnings)


def test_start_in_inflation_beyond_budget_fails():
    g = grid_with_block(block=(1.0, 1.0, 5.0, 5.0))
    # deep inside a big block, no free cell within max_shift
    r = plan(g, (3.0, 3.0), (5.5, 5.5), PlanConfig(r_circ=0.3, max_shift=0.3))
    assert r.success is False


def test_acceptance8_deterministic_bit_identical():
    """acceptance 8: same input twice -> identical path (invariant 6)."""
    g = grid_with_block()
    cfg = PlanConfig(r_circ=0.3, soft_weight=2.0)
    r1 = plan(g, (0.5, 0.5), (5.5, 5.5), cfg)
    r2 = plan(g, (0.5, 0.5), (5.5, 5.5), cfg)
    assert r1.success and r2.success
    assert np.array_equal(r1.path.points, r2.path.points)
    assert r1.expanded_nodes == r2.expanded_nodes


def test_expanded_nodes_reported():
    g = empty_grid()
    r = plan(g, (0.5, 0.5), (5.0, 5.0))
    assert r.expanded_nodes > 0


# --------------------------------------------------------------------------- #
# shortcut
# --------------------------------------------------------------------------- #
def test_acceptance9_shortcut_stays_collision_free():
    """acceptance 9: the shortcut path crosses no lethal cell."""
    g = grid_with_block()
    cf = CostField(g, PlanConfig(r_circ=0.3))
    r = GridPlanner(cf).plan((0.5, 0.5), (5.5, 5.5))
    sc = shortcut_path(cf, r.path)
    # every segment endpoint pair must have line of sight on the lethal grid
    for a, b in zip(sc.points[:-1], sc.points[1:]):
        assert line_of_sight(cf, a, b)


def test_acceptance10_shortcut_not_longer():
    """acceptance 10: shortcut length <= original length."""
    g = grid_with_block()
    cf = CostField(g, PlanConfig(r_circ=0.3))
    r = GridPlanner(cf).plan((0.5, 0.5), (5.5, 5.5))
    sc = shortcut_path(cf, r.path)
    assert sc.length <= r.path.length + 1e-9
    assert len(sc) <= len(r.path)


def test_line_of_sight_blocked_by_wall():
    g = grid_with_block()
    cf = CostField(g, PlanConfig(r_circ=0.2))
    # straight line through the block centre must be blocked
    assert not line_of_sight(cf, (1.0, 3.0), (5.0, 3.0))
    # clear line in open space
    assert line_of_sight(cf, (0.5, 0.5), (0.5, 5.5))


# --------------------------------------------------------------------------- #
# corners
# --------------------------------------------------------------------------- #
def test_acceptance11_consecutive_corners_arcs_do_not_overlap():
    """acceptance 11: clamped tangent offsets <= half the shorter adjacent leg."""
    # zig-zag with short legs and sharp 90 deg turns
    pts = np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 0.3],
                    [0.6, 0.3], [0.6, 0.0]])
    off = tangent_offsets(pts, delta=0.05)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    for i in range(1, len(pts) - 1):
        half_min = 0.5 * min(seg[i - 1], seg[i])
        assert off[i] <= half_min + 1e-9


def test_corner_speed_straight_is_unlimited_uturn_is_zero():
    straight = np.array([[0, 0], [1, 0], [2, 0]])
    assert np.isinf(corner_speed_limits(straight, a_max=1.0)[1])
    uturn = np.array([[0, 0], [1, 0], [0, 0.0001]])
    R, phi = corner_radii(uturn)
    assert phi[1] > np.pi / 2  # a near-U-turn
    assert corner_speed_limits(uturn, a_max=1.0)[1] < 0.5


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def test_acceptance12_theta_continuous_across_pi():
    """acceptance 12: heading is continuous (unwrapped) across the +-pi cut."""
    # a path that, aimed at a target, sweeps heading through +-pi
    pts = np.array([[-1.0, 0.01], [-0.5, 0.0], [0.0, -0.01], [0.5, 0.0], [1.0, 0.01]])
    from omni_sim_core.planning.types import Path
    prof = build_orientation_profile(Path(pts), mode="target", target=(0.0, 5.0))
    # no single step jumps by more than pi -> unwrapped
    assert np.all(np.abs(np.diff(prof.theta)) < np.pi)


def test_orientation_modes():
    from omni_sim_core.planning.types import Path
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    fx = build_orientation_profile(Path(pts), mode="fixed", theta=0.7)
    assert np.allclose(fx.theta, 0.7)
    wp = build_orientation_profile(Path(pts), mode="waypoint",
                                   thetas=[0.0, 0.5, 1.0])
    assert wp.theta[0] == 0.0 and wp.theta[-1] == 1.0
    assert fx.s[0] == 0.0 and fx.s[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# performance gate (checkpoint: < 1 s / plan)
# --------------------------------------------------------------------------- #
def test_plan_under_one_second_on_full_field():
    """Checkpoint 3: a 240x240 grid must plan in well under a second."""
    g = generate_rect_field(12.0, 12.0, 0.05)  # 240 x 240
    t0 = time.perf_counter()
    r = plan(g, (0.5, 0.5), (11.5, 11.5), PlanConfig(r_circ=0.3))
    dt = time.perf_counter() - t0
    assert r.success
    assert dt < 1.0, f"plan took {dt:.3f}s (>1s budget)"


# --------------------------------------------------------------------------- #
# multi-level transition waypoints vs the real chassis radius
# --------------------------------------------------------------------------- #

def test_transition_waypoints_are_plannable_for_the_real_chassis():
    """Every stitched-plan waypoint must be non-lethal on its OWN level's grid.

    These four points are the only thing holding a multi-level plan together:
    A* runs per level, and each leg's endpoint is one of them. A waypoint that
    drifts into inflation doesn't degrade gracefully -- the whole plan fails
    with "goal shifted out of inflation" and the robot never moves.

    They were originally hand-tuned against r_circ=0.45. The real 0.9 m square
    chassis circumscribes at 0.636 m, which moved every free band and made all
    four lethal at once; the corridors are narrow enough (the L1 ring is 1.5 m
    wide, leaving 0.228 m after inflating both sides) that this is a real risk
    on any chassis or field-spec change. scripts/measure_corridors.py prints
    the bands these have to sit in.
    """
    from pathlib import Path

    from omni_sim_core.field import LayeredField
    from omni_sim_core.ui.leveled_field_2027 import (_RAMP_WAYPOINTS_M,
                                                     _STAIRS_WAYPOINTS_M)

    root = Path(__file__).resolve().parents[2]
    field = LayeredField.from_yaml(root / "config" / "field" / "robocon2027.yaml")
    r_circ = 0.7 * np.sqrt(2) / 2                      # config/robot/chassis.yaml
    cfields = {lvl: CostField(field.grid(lvl, "nav"), PlanConfig(r_circ=r_circ))
               for lvl in ("ground", "l1", "l2")}

    def assert_free(level, pt, what):
        g = field.grid(level, "nav")
        r, c = g.world_to_grid(*pt)
        assert not cfields[level].lethal[r, c], (
            f"{what} {pt} is lethal on {level} at r_circ={r_circ:.3f} -- "
            "re-measure with scripts/measure_corridors.py")

    ground_pt, l1_pt = _RAMP_WAYPOINTS_M["red"]
    assert_free("ground", ground_pt, "red ramp ground-side waypoint")
    assert_free("l1", l1_pt, "red ramp L1-side waypoint")

    ground_pt, l1_pt = _RAMP_WAYPOINTS_M["blue"]
    assert_free("ground", ground_pt, "blue ramp ground-side waypoint")
    assert_free("l1", l1_pt, "blue ramp L1-side waypoint")

    l1_pt, l2_pt = _STAIRS_WAYPOINTS_M
    assert_free("l1", l1_pt, "stairs L1-side waypoint")
    assert_free("l2", l2_pt, "stairs L2-side waypoint")


def test_l1_ring_is_reachable_from_the_ramp_and_reaches_the_stairs():
    """The ground->L1->L2 chain exists as *connected free space*, not just as
    four waypoints that each happen to be free.

    The L1 ring is the tight link: 1.5 m between the L1 and L2 slab edges, and
    the centre divider fence stub sticks into it, so the reachable part stops
    at the Stairs gate's western sliver. If it stops any earlier, L2 becomes
    unreachable for this chassis and the multi-level planner is decorative.
    """
    from pathlib import Path

    from scipy.ndimage import label

    from omni_sim_core.field import LayeredField
    from omni_sim_core.ui.leveled_field_2027 import (_RAMP_WAYPOINTS_M,
                                                     _STAIRS_WAYPOINTS_M)

    root = Path(__file__).resolve().parents[2]
    field = LayeredField.from_yaml(root / "config" / "field" / "robocon2027.yaml")
    g = field.grid("l1", "nav")
    cf = CostField(g, PlanConfig(r_circ=0.7 * np.sqrt(2) / 2))

    comp, _ = label(~cf.lethal)
    r, c = g.world_to_grid(*_RAMP_WAYPOINTS_M["red"][1])   # where the ramp lands
    ramp_side = comp[r, c]
    assert ramp_side != 0, "the ramp exit itself is lethal on L1"

    r, c = g.world_to_grid(*_STAIRS_WAYPOINTS_M[0])        # foot of the stairs
    assert comp[r, c] == ramp_side, (
        "the L1 stairs waypoint is not reachable from the ramp exit -- "
        "L2 is cut off for this chassis")


def test_the_planner_grid_refuses_the_opponents_half():
    """A* must not offer a route the collision rule will then refuse.

    Rule 6.2.2 lives in ``LeveledField.is_blocked``, which the raster knows
    nothing about, so for a while the two disagreed: asked for a goal whose
    shortest route crossed the centre line, the planner returned a 29 m path
    around the outside of the field through the blue half, the robot followed
    it for a few metres and jammed against a wall in no grid -- with the plan
    still drawn on screen showing the way "through". The planner grid carries
    the rule now; this pins that down.
    """
    from pathlib import Path

    from omni_sim_core.env.occupancy_grid import OCCUPIED
    from omni_sim_core.ui.leveled_field_2027 import LeveledField

    maps = Path(__file__).resolve().parents[2] / "maps"
    yamls = [maps / f"field_2027_{lvl}.yaml" for lvl in ("ground", "l1", "l2")]
    if not all(p.exists() for p in yamls):
        pytest.skip("generated maps/ not present")

    lf = LeveledField(*(str(p) for p in yamls), team="red")
    for level in ("ground", "l1"):
        g = lf.territory_masked_grid(level)
        # deep in the opponent's half, far from any wall -> occupied for us
        r, c = g.world_to_grid(9.5, 5.0)
        assert g.grid[r, c] == OCCUPIED, f"{level}: blue half is plannable"
        # ...and every free cell left is somewhere we are allowed to be
        for row in range(0, g.height, 17):
            for col in range(0, g.width, 17):
                if g.grid[row, col] != OCCUPIED:
                    x, y = g.grid_to_world(row, col)
                    assert lf.territory_ok(level, x, y), (
                        f"{level}: ({x:.2f}, {y:.2f}) is plannable but "
                        "territory_ok() would refuse it")

    # the Ground Shared Area straddles the centre line and stays open
    g = lf.territory_masked_grid("ground")
    r, c = g.world_to_grid(5.6, 1.0)
    assert g.grid[r, c] != OCCUPIED, "the Ground Shared Area was masked away"

    # L2 is one Shared Area -- masking it would cut the field in half for no
    # reason, so it must come back untouched (identically, not just equal)
    assert lf.territory_masked_grid("l2") is lf.grids["l2"]


def test_gate_waypoint_pairs_are_two_distinct_points():
    """The two sides of a gate are a gap apart, so a stitched multi-level plan
    must keep *both*.

    sim_node concatenates one A* leg per level and used to drop each new leg's
    first point as "the point shared with the previous leg". Legs are not
    contiguous across a gate: leg k ends on the *from* side and leg k+1 starts
    on the *to* side, with the un-plannable slope in between. Dropping it made
    the followed path run from one side of the gate straight to the next free
    vertex beyond it -- a diagonal that leaves the gate rectangle, where no
    level is in play, so the robot jammed half-way through the transition and
    the follower then quietly gave up. Nothing here shares a point: check it.
    """
    from pathlib import Path

    from omni_sim_core.ui.leveled_field_2027 import LeveledField

    maps = Path(__file__).resolve().parents[2] / "maps"
    yamls = [maps / f"field_2027_{lvl}.yaml" for lvl in ("ground", "l1", "l2")]
    if not all(p.exists() for p in yamls):
        pytest.skip("generated maps/ not present")

    for team in ("red", "blue"):
        lf = LeveledField(*(str(p) for p in yamls), team=team)
        for a, b in (("ground", "l1"), ("l1", "l2")):
            for lo, hi in ((a, b), (b, a)):            # both directions
                p, q = lf.transition_waypoint_pair(lo, hi)
                gap = np.hypot(q[0] - p[0], q[1] - p[1])
                assert gap > lf.grids[lo].meta.resolution, (
                    f"{team} {lo}->{hi}: the gate's two waypoints are {gap:.3f} m "
                    "apart -- if they ever coincide, revisit the de-duplication "
                    "in sim_node._on_goal_pose")
