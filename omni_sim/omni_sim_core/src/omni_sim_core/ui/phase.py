"""Two small state machines to display: what the robot is doing, and where.

Deliberately kept to what can be *derived* rather than declared:

``ControlState``  the executive's own state -- idle, planning, following,
                  held back, blocked, finished. Every one of these is already
                  implied by signals sim_node has; naming them just stops the
                  dashboard from re-deriving the same thing slightly wrong.

``MatchPhase``    where on the field the robot is, in task terms. This is a
                  *location* machine, not a game model: nothing here knows
                  about scoring, possession or a Sky Block, because none of
                  that is modelled yet. Calling it a match phase when it only
                  reads coordinates would be overclaiming, so the topic
                  carries it as ``phase`` and the labels are place names.

Both are plain strings over the wire. The transition *history* is what makes
them worth showing -- a current value is a readout, a sequence is a trace --
so ``PhaseTracker`` keeps a short one and only records genuine changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .field_layout_2027 import (
    CENTER_X_MM, GROUND_SHARED_RECT_MM, L1_RETRY_RECT_MM, L1_SHARED_STRIP_MM,
    MUSTIKA_SHARED_RECT_MM, RAMP_GATE_RECT_MM, START_RECTS_MM,
    STAIRS_GATE_MARGIN_MM, STORAGE_RECT_MM, TRANSFER_RECT_MM, _mirror,
)

# The order they are expected to happen in, which is also the order a strip of
# chips should be drawn in. Not enforced -- a robot may skip or go backwards.
CONTROL_STATES = ("idle", "planning", "following", "held", "blocked",
                  "arrived", "gave_up", "manual")
MATCH_PHASES = ("start", "ground", "transfer", "storage", "ramp", "l1",
                "stairs", "l2", "shared", "retry")


def _in(x: float, y: float, rect) -> bool:
    x0, y0, x1, y1 = (v / 1000.0 for v in rect)
    return x0 <= x <= x1 and y0 <= y <= y1


def match_phase(x: float, y: float, level: str, team: str = "red") -> str:
    """Where the robot is, in task terms. Metres, world frame.

    Order matters: the gates are tested before the levels they connect,
    because a robot on the Ramp is on neither floor and saying "ground" or
    "l1" there would hide the most interesting few seconds of the run.
    """
    def mine(rect):
        return rect if team == "red" else _mirror(rect)

    if _in(x, y, mine(RAMP_GATE_RECT_MM)):
        return "ramp"
    if _in(x, y, STAIRS_GATE_MARGIN_MM) and level in ("l1", "l2"):
        return "stairs"
    if level == "l2":
        return "l2"
    if any(_in(x, y, mine(r)) for r in START_RECTS_MM):
        return "start"
    if _in(x, y, mine(STORAGE_RECT_MM)):
        return "storage"
    if _in(x, y, mine(TRANSFER_RECT_MM)):
        return "transfer"
    if level == "l1":
        if _in(x, y, mine(L1_RETRY_RECT_MM)):
            return "retry"
        if _in(x, y, L1_SHARED_STRIP_MM):
            return "shared"
        return "l1"
    if _in(x, y, GROUND_SHARED_RECT_MM) or _in(x, y, MUSTIKA_SHARED_RECT_MM):
        return "shared"
    return "ground"


def control_state(*, following: bool, planning: bool, blocked: bool,
                  held: bool, manual: bool, last_outcome: str | None) -> str:
    """The executive's state, in the order the checks have to be made.

    ``blocked`` outranks ``following`` on purpose: "following" while pinned
    against a wall is the reading that made the old silent-stall so hard to
    see. ``held`` is the governed follower waiting for the robot to catch up
    -- a state the trajectory code has had since the reference governor went
    in, but which nothing displayed.
    """
    if blocked:
        return "blocked"
    if planning:
        return "planning"
    if held:
        return "held"
    if following:
        return "following"
    if manual:
        return "manual"
    if last_outcome in ("arrived", "gave_up"):
        return last_outcome
    return "idle"


@dataclass
class PhaseTracker:
    """Current value plus a short history of genuine changes."""

    keep: int = 12
    current: str | None = None
    history: list[tuple[float, str]] = field(default_factory=list)

    def update(self, t: float, value: str) -> bool:
        """Record ``value`` at time ``t``. True if it was a transition."""
        if value == self.current:
            return False
        self.current = value
        self.history.append((round(float(t), 2), value))
        del self.history[:-self.keep]
        return True

    def as_dict(self) -> dict:
        return {"current": self.current,
                "history": [[t, v] for t, v in self.history]}
