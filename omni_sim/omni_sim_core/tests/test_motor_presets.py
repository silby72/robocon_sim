"""config/presets/motors/*.yaml -- the numbers a person types in by hand.

These files are transcribed from manufacturer datasheets. Nothing checks them,
and half of what they declare is never read: ``derive_motor`` computes the
stall torque from ``Kt * (I_stall - I_noload)`` and the no-load speed from
``kv * V``, so the ``stall_torque_nm`` and ``no_load_speed_rpm`` fields sit in
the file looking authoritative and change nothing.

That is fine as long as they *agree* with what the model derives -- then they
read as a cross-check. Where they disagree, the file is lying to the next
person to open it, and the size of the lie is not small: ak40_10_v3 claims
24.5 Nm and the simulator drives with 5.6.

So the rule these tests enforce is: **every declared number must either be used
or agree with the model**. A field that is neither is a trap.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import yaml

from omni_sim_core.mechanism.derive import derive_motor
from omni_sim_core.mechanism.schema import Datasheet, Physical

ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = ROOT / "config" / "presets" / "motors"

pytestmark = pytest.mark.skipif(not PRESET_DIR.exists(),
                                reason="repo config/ not available")

# How far a declared value may sit from the derived one before it counts as a
# contradiction rather than rounding. Datasheets round hard -- m3508 publishes
# 4.77 against a derived 4.7746 -- so this is generous on purpose.
TORQUE_TOL = 0.05      # 5 %
SPEED_TOL = 0.02       # 2 %

# --------------------------------------------------------------------------- #
# Presets whose declared stall torque contradicts the model, recorded with the
# ratio as measured today. They are NOT approved: this is the note that cannot
# get lost. Fixing one makes its test fail, which is the point -- delete the
# entry then, and say in the file's header comment which shaft the numbers are
# referred to.
#
#   m2006_c610   1.49x  declared 1.0 Nm, model 1.49. DJI publishes ~1.0 Nm
#                       continuous for the M2006 output shaft; the model's
#                       figure is a *stall* torque from the current limit, so
#                       these are two different quantities sharing a field.
#   generic_dc   0.48x  a made-up placeholder motor; its numbers were never
#                       meant to be self-consistent.
#   ak40_10_v3   0.23x  mixes shaft references. No Kt satisfies both kv=100
#                       rpm/V and 24.5 Nm at 60 A (that pair implies Kt=0.417,
#                       i.e. kv=22.9). The AK70-10 has a ~10:1 planetary and
#                       the file has the rotor-side kv with the output-side
#                       torque. Someone with the real datasheet has to pick.
# --------------------------------------------------------------------------- #
KNOWN_INCONSISTENT_TORQUE = {
    "m2006_c610": 1.49,
    "generic_dc": 0.48,
    "ak40_10_v3": 0.23,
}

REQUIRED = ["rated_voltage_v", "kv_rpm_per_v", "stall_torque_nm",
            "stall_current_a", "no_load_speed_rpm", "no_load_current_a",
            "encoder_cpr"]


def preset_files():
    return sorted(PRESET_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _derive(path: Path):
    doc = _load(path)
    ds = Datasheet.from_doc(doc["datasheet"], path.stem)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the rotor-inertia estimate
        return ds, derive_motor(ds, Physical.from_doc(doc.get("physical")),
                                gear_ratio=1.0,
                                load_inertia_wheel_side_kgm2=0.0)


def ids():
    return [p.stem for p in preset_files()]


@pytest.fixture(params=preset_files(), ids=ids())
def preset(request):
    return request.param


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
def test_there_are_presets_to_check():
    assert preset_files(), "no motor presets found; the rest of this file is vacuous"


def test_it_parses_and_has_every_required_field(preset):
    doc = _load(preset)
    assert "datasheet" in doc, f"{preset.name}: no 'datasheet' block"
    missing = [k for k in REQUIRED if doc["datasheet"].get(k) is None]
    assert not missing, f"{preset.name}: missing {missing}"


def test_every_quantity_is_positive(preset):
    ds = _load(preset)["datasheet"]
    for k in REQUIRED:
        assert float(ds[k]) > 0, f"{preset.name}: {k} must be > 0, got {ds[k]}"
    if ds.get("rotor_inertia_kgm2") is not None:
        assert float(ds["rotor_inertia_kgm2"]) > 0, preset.name


def test_the_no_load_current_is_below_the_stall_current(preset):
    """Otherwise the derived stall torque comes out negative and the motor
    drives backwards -- silently, since nothing downstream checks the sign."""
    ds = _load(preset)["datasheet"]
    assert float(ds["no_load_current_a"]) < float(ds["stall_current_a"]), \
        f"{preset.name}: no-load current is not below stall current"


def test_it_says_where_the_numbers_came_from(preset):
    """A preset with no provenance comment cannot be checked against anything
    later. The header is the only record of which datasheet, and -- for a
    gearmotor -- which shaft the numbers are referred to."""
    head = preset.read_text(encoding="utf-8").split("datasheet:")[0]
    lines = [ln for ln in head.splitlines() if ln.strip().startswith("#")]
    assert sum(len(ln) for ln in lines) > 40, (
        f"{preset.name}: no header comment saying where these numbers come from")


def test_the_rotor_inertia_is_either_given_or_deliberately_null(preset):
    """Omitting the key and writing ``null`` mean the same thing to the schema
    but not to a reader: null with a comment is a decision, a missing key is
    an oversight. derive.py warns and flags ESTIMATED either way."""
    ds = _load(preset)["datasheet"]
    assert "rotor_inertia_kgm2" in ds, (
        f"{preset.name}: rotor_inertia_kgm2 absent. Write it out as null if it "
        "is genuinely unknown, so the estimate is a choice and not an accident")


# --------------------------------------------------------------------------- #
# the declared values against the model
# --------------------------------------------------------------------------- #
def test_the_declared_no_load_speed_matches_the_one_the_model_uses(preset):
    """``no_load_speed_rpm`` is never read: the model uses ``kv * V``. It is
    worth keeping only as a cross-check, which means it has to agree."""
    ds, m = _derive(preset)
    model_rpm = m.no_load_speed_rad_s * 60.0 / (2.0 * math.pi)
    rel = abs(model_rpm - ds.no_load_speed_rpm) / ds.no_load_speed_rpm
    assert rel <= SPEED_TOL, (
        f"{preset.stem}: declares {ds.no_load_speed_rpm:.0f} rpm but kv*V gives "
        f"{model_rpm:.0f} rpm ({rel:.0%} off). The simulator uses the latter; "
        "one of the two numbers is wrong")


def test_the_declared_stall_torque_matches_the_one_the_model_uses(preset):
    """``stall_torque_nm`` is never read either: the model uses
    ``Kt * (I_stall - I_noload)``. Same argument, and this is where the
    presets actually disagree -- see KNOWN_INCONSISTENT_TORQUE."""
    ds, m = _derive(preset)
    ratio = m.stall_torque_nm / ds.stall_torque_nm
    known = KNOWN_INCONSISTENT_TORQUE.get(preset.stem)
    if known is not None:
        assert ratio == pytest.approx(known, abs=0.01), (
            f"{preset.stem}: the inconsistency changed (was {known:.2f}x, now "
            f"{ratio:.2f}x). If it was fixed, drop it from "
            "KNOWN_INCONSISTENT_TORQUE and say in the file header which shaft "
            "the numbers are referred to")
        return
    assert abs(ratio - 1.0) <= TORQUE_TOL, (
        f"{preset.stem}: declares {ds.stall_torque_nm:.3f} Nm but "
        f"Kt*(I_stall-I_noload) gives {m.stall_torque_nm:.3f} Nm ({ratio:.2f}x). "
        "The simulator uses the latter. Fix the datasheet block, or -- if the "
        "motor really is like this -- add it to KNOWN_INCONSISTENT_TORQUE with "
        "a note saying why")


def test_a_new_preset_cannot_quietly_join_the_inconsistent_list():
    """The allowlist is for the three presets that were already wrong when it
    was written. It is not a place to put new ones."""
    assert set(KNOWN_INCONSISTENT_TORQUE) == {"m2006_c610", "generic_dc",
                                              "ak40_10_v3"}, (
        "KNOWN_INCONSISTENT_TORQUE changed. Adding a motor whose declared "
        "torque disagrees with its own kv and current ratings means one of "
        "those three numbers is a transcription error -- find which, rather "
        "than recording the discrepancy")


def test_every_preset_is_reachable_from_the_browser_bundle():
    """The chassis panel offers whatever the bundle holds. A preset added to
    config/ and not synced is invisible there, and the difference looks like
    the page being broken rather than stale."""
    bundle = ROOT / "omni_sim_web" / "js" / "core-bundle.js"
    if not bundle.exists():
        pytest.skip("web bundle not generated")
    text = bundle.read_text(encoding="utf-8")
    for p in preset_files():
        assert f'"{p.stem}"' in text, (
            f"{p.stem} is not in the browser bundle -- run scripts/sync_web_core.sh")
