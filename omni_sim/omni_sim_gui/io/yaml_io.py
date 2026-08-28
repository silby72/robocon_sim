"""Comment-preserving chassis.yaml IO for the GUI.

Wraps ``omni_sim_core.mechanism.yaml_rt`` (ruamel) so that editing a few values
in the GUI and saving preserves the author's comments and key order. All values
written to the document are SI; the GUI works in mm/degrees and converts here.

The in-memory document is the ruamel ``CommentedMap`` itself: the GUI mutates
scalar leaves in place, so unchanged parts round-trip byte-for-byte.
"""
from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import Any

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from omni_sim_core.mechanism.yaml_rt import rt_dump, rt_dumps, rt_load, rt_loads
from omni_sim_core.mechanism.schema import Chassis

# -- display-unit conversions (kept strictly inside the GUI) ----------------
def m_to_mm(x_m: float) -> float:
    return x_m * 1000.0


def mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0


def rad_to_deg(x_rad: float) -> float:
    return math.degrees(x_rad)


def deg_to_rad(x_deg: float) -> float:
    return math.radians(x_deg)


def load_chassis_doc(path: str | Path):
    """Load chassis.yaml as a ruamel document (comments/order preserved)."""
    return rt_load(path)


def loads_doc(text: str):
    """Parse a YAML string back into a ruamel document (for undo snapshots)."""
    return rt_loads(text)


def load_actuators_doc(path: str | Path):
    """Load actuators.yaml as a ruamel document (comments/order preserved)."""
    return rt_load(path)


def new_actuator(preset: str | None) -> CommentedMap:
    """A fresh actuator entry: a preset reference with an empty override map."""
    m = CommentedMap()
    m["preset"] = preset
    m["overrides"] = CommentedMap()
    return m


def load_actuator_names(path: str | Path) -> list[str]:
    """Return the actuator ids defined in actuators.yaml (for the ref dropdown).

    Missing/unreadable file -> empty list (the dropdown still lets you type a
    ref; nothing is flagged as dangling when the actuator set is unknown).
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        doc = rt_load(path)
        return [str(k) for k in (doc.get("actuators", {}) or {}).keys()]
    except Exception:
        return []


def _flow_seq(values: list) -> CommentedSeq:
    seq = CommentedSeq(values)
    seq.fa.set_flow_style()          # keep [x, y] inline like the authored file
    return seq


def new_wheel(wid: str, x_m: float, y_m: float, theta_rad: float,
              radius_m: float, gear_ratio: float, actuator_ref: str) -> CommentedMap:
    """A ruamel mapping for a fresh drive wheel (SI units)."""
    m = CommentedMap()
    m["id"] = wid
    m["position_m"] = _flow_seq([round(x_m, 6), round(y_m, 6)])
    m["drive_axis_rad"] = round(theta_rad, 6)
    m["radius_m"] = radius_m
    m["gear_ratio"] = gear_ratio
    m["actuator_ref"] = actuator_ref
    m["reverse"] = False
    return m


def new_odometry(oid: str, x_m: float, y_m: float, measure_axis_rad: float,
                 radius_m: float, encoder_cpr: int) -> CommentedMap:
    """A ruamel mapping for a fresh standalone dead-wheel odometry unit (SI)."""
    m = CommentedMap()
    m["id"] = oid
    m["type"] = "dead_wheel"
    m["position_m"] = _flow_seq([round(x_m, 6), round(y_m, 6)])
    m["measure_axis_rad"] = round(measure_axis_rad, 6)
    m["radius_m"] = radius_m
    m["encoder_cpr"] = encoder_cpr
    return m


def parse_chassis(doc) -> Chassis:
    """Validate/parse the document into the core dataclass (for jacobian etc.)."""
    return Chassis.from_doc(doc)


def diff_text(original: str, updated: str) -> str:
    """Unified diff shown to the user before saving (spec 6.4)."""
    lines = difflib.unified_diff(
        original.splitlines(), updated.splitlines(),
        fromfile="chassis.yaml (on disk)", tofile="chassis.yaml (to write)",
        lineterm="")
    return "\n".join(lines)


def render_doc(doc) -> str:
    return rt_dumps(doc)


def save_chassis_doc(doc, path: str | Path) -> None:
    """Atomically write the document back (temp file + os.replace)."""
    rt_dump(doc, path)
