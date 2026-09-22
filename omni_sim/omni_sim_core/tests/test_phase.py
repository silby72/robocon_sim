"""The two displayed state machines.

Small on purpose: both are derived, not declared, so the only things worth
pinning are that the derivation matches the field geometry it claims to read,
and that the tracker records a *trace* rather than a sampled readout.
"""
from __future__ import annotations

import pytest

from omni_sim_core.ui.phase import (CONTROL_STATES, MATCH_PHASES, PhaseTracker,
                                    control_state, match_phase)


def test_the_phase_follows_a_ground_to_l2_traverse():
    """The sequence a multi-level run must produce, in order."""
    waypoints = [
        ((0.65, 0.65), "ground", "start"),
        ((1.00, 4.75), "ground", "ground"),
        ((2.00, 4.50), "ground", "ground"),    # part-way up: no gate needed
        ((2.00, 6.20), "ground", "ramp"),      # the landing, inside the Ramp gate
        ((3.60, 4.75), "l1", "l1"),      # clear of the gate (see below)
        ((4.80, 4.00), "l1", "stairs"),        # inside the Stairs gate
        ((5.50, 6.00), "l2", "l2"),
    ]
    for (x, y), level, expected in waypoints:
        assert match_phase(x, y, level) == expected, f"at ({x}, {y}) on {level}"


def test_the_ramp_phase_covers_the_landing_not_the_slope():
    """The gate is the *landing* plus a margin east onto the slab, because
    that short move is the only part where both levels are in play: the sloped
    section is cleared on the ground grid alone, so a robot part-way up is
    simply on the ground.

    The phase reuses the functional gate rather than the painted ramp, so what
    is displayed as "mid-transition" is the same region the collision check
    treats as mid-transition."""
    assert match_phase(2.00, 4.50, "ground") == "ground"   # on the slope
    assert match_phase(2.00, 6.20, "ground") == "ramp"     # on the landing
    assert match_phase(3.00, 6.20, "l1") == "ramp"         # stepping onto the slab
    assert match_phase(3.60, 6.20, "l1") == "l1"           # clear of the gate


def test_a_gate_outranks_the_levels_it_connects():
    """Inside the gate the robot is mid-transition; reporting a level there
    would hide the most interesting seconds of the run."""
    assert match_phase(2.0, 6.2, "ground") == "ramp"
    assert match_phase(2.0, 6.2, "l1") == "ramp"


def test_phases_mirror_for_the_blue_team():
    """Red's Ramp is at x~2; blue's is its mirror, and neither team should see
    the other's furniture as its own."""
    assert match_phase(2.0, 6.2, "ground", team="red") == "ramp"
    assert match_phase(2.0, 6.2, "ground", team="blue") != "ramp"
    assert match_phase(9.0, 6.2, "ground", team="blue") == "ramp"


def test_every_derived_phase_is_one_the_ui_draws():
    """A phase the dashboard has no chip for would silently vanish."""
    seen = set()
    for level in ("ground", "l1", "l2"):
        for x in [i / 10 for i in range(5, 110, 3)]:
            for y in [i / 10 for i in range(5, 110, 3)]:
                seen.add(match_phase(x, y, level))
    assert seen <= set(MATCH_PHASES), f"undrawable phases: {seen - set(MATCH_PHASES)}"


@pytest.mark.parametrize("kwargs,expected", [
    (dict(following=False, planning=False, blocked=False, held=False,
          manual=False, last_outcome=None), "idle"),
    (dict(following=True, planning=False, blocked=False, held=False,
          manual=False, last_outcome=None), "following"),
    # blocked outranks following: "following" while pinned against a wall is
    # exactly the reading that made the old silent stall so hard to see
    (dict(following=True, planning=False, blocked=True, held=True,
          manual=False, last_outcome=None), "blocked"),
    (dict(following=True, planning=False, blocked=False, held=True,
          manual=False, last_outcome=None), "held"),
    (dict(following=False, planning=False, blocked=False, held=False,
          manual=False, last_outcome="gave_up"), "gave_up"),
])
def test_control_state_priority(kwargs, expected):
    assert control_state(**kwargs) == expected
    assert expected in CONTROL_STATES


def test_the_tracker_records_transitions_not_samples():
    tr = PhaseTracker(keep=3)
    assert tr.update(0.0, "idle") is True
    assert tr.update(0.1, "idle") is False      # same value, not a transition
    for t, v in ((1.0, "planning"), (1.1, "following"), (9.0, "arrived")):
        assert tr.update(t, v) is True
    assert tr.current == "arrived"
    # keep=3 drops the oldest, never the newest
    assert [v for _, v in tr.history] == ["planning", "following", "arrived"]
    assert tr.as_dict()["current"] == "arrived"
