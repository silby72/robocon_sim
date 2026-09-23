"""The browser bundle is a copy, and copies rot. This is the gate.

Deliberately *not* under ``omni_sim_core/tests/``: the brief forbids touching
that tree, and this tests the web mirror rather than the core. Run it with the
rest::

    .venv/bin/python -m pytest omni_sim_web/tests/test_web_bundle.py

What can go wrong with a copy, in the order it actually happens:

1. Someone edits ``mechanism/derive.py`` and does not re-run the sync. The page
   keeps serving yesterday's physics and looks completely healthy.
2. Someone edits the copy in ``omni_sim_web/py/`` -- "just a small fix for the
   browser" -- and the two implementations quietly diverge. This is the failure
   the whole copy-verbatim rule exists to prevent, so it is a test, not a note.
3. The golden values go stale against a changed core, and ``verify.html`` turns
   red for a reason nobody can find from the browser alone.

Each gets a test, and each failure message says which script to re-run.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "omni_sim_web"
SRC = ROOT / "omni_sim_core" / "src"

sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not (WEB / "js" / "core-bundle.js").exists(),
    reason="web bundle not generated; run scripts/sync_web_core.sh")

RESYNC = "\n\nre-run: scripts/sync_web_core.sh"
REGOLD = "\n\nre-run: .venv/bin/python scripts/export_golden_values.py"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _json_after(path: Path, marker: str) -> dict:
    """Pull the JSON object out of a ``window.X = {...};`` file."""
    text = path.read_text(encoding="utf-8")
    start = text.index(marker) + len(marker)
    return json.loads(text[start:].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def bundle() -> dict:
    return _json_after(WEB / "js" / "core-bundle.js", "window.OMNI_SIM_CORE = ")


@pytest.fixture(scope="module")
def golden() -> dict:
    return _json_after(WEB / "js" / "golden.js", "window.OMNI_SIM_GOLDEN = ")


# --------------------------------------------------------------------------- #
# the copy is a copy
# --------------------------------------------------------------------------- #
def test_the_bundle_matches_the_core_it_was_copied_from(bundle):
    """Failure means omni_sim_core moved and the browser did not follow."""
    stale = []
    for rel, text in bundle["modules"].items():
        src = SRC / rel
        assert src.exists(), f"{rel} is in the bundle but gone from the core"
        if src.read_text(encoding="utf-8") != text:
            stale.append(rel)
    assert not stale, f"stale in the browser bundle: {stale}{RESYNC}"


def test_the_readable_copy_matches_the_bundle(bundle):
    """omni_sim_web/py/ is for reading and diffing; if it disagrees with what
    actually ships, it is worse than not existing -- a reviewer would check the
    copy they can read and approve a bundle that says something else."""
    wrong = []
    for rel, text in bundle["modules"].items():
        copy = WEB / "py" / rel
        assert copy.exists(), f"{rel} missing from omni_sim_web/py/{RESYNC}"
        if copy.read_text(encoding="utf-8") != text:
            wrong.append(rel)
    assert not wrong, f"omni_sim_web/py/ differs from the shipped bundle: {wrong}{RESYNC}"


def test_the_recorded_hashes_are_the_hashes(bundle):
    for rel, text in bundle["modules"].items():
        assert bundle["hashes"][rel] == _sha(text), f"hash lies about {rel}"


def test_the_module_list_is_what_the_sync_script_declares(bundle):
    from sync_web_core import MODULES
    assert sorted(bundle["modules"]) == sorted(MODULES), (
        "the bundle and the sync script disagree about what ships" + RESYNC)


def test_no_module_pulls_in_something_that_cannot_reach_the_browser(bundle):
    """The page loads in under two seconds because the copied slice imports
    almost nothing. An import of yaml, scipy, or the simulator would not fail
    loudly in the browser -- it would fail at first use, on a visitor's
    machine, in a stack trace nobody sees."""
    allowed = {"math", "warnings", "dataclasses", "typing", "logging",
               "numpy", "__future__", "collections", "enum"}
    for rel, text in bundle["modules"].items():
        for line in text.splitlines():
            m = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w.]*)", line)
            if not m:
                continue
            top = m.group(1).split(".")[0]
            if top in allowed or line.lstrip().startswith(("from .", "import .")):
                continue
            pytest.fail(f"{rel} imports {top!r}, which does not ship: {line.strip()}")


# --------------------------------------------------------------------------- #
# the presets
# --------------------------------------------------------------------------- #
def test_every_motor_preset_shipped(bundle):
    import yaml
    on_disk = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
               for p in sorted((ROOT / "config" / "presets" / "motors").glob("*.yaml"))}
    assert bundle["presets"] == on_disk, (
        "the browser's motor presets are not the repo's" + RESYNC)


# --------------------------------------------------------------------------- #
# the golden values
# --------------------------------------------------------------------------- #
def test_the_golden_values_still_describe_this_core(golden):
    """The check verify.html cannot make: recompute every expected value here,
    in the environment the rest of the suite runs in. If this passes and the
    browser disagrees, the difference is Pyodide's -- which is the one thing
    the browser test is actually able to tell us."""
    from export_golden_values import build
    fresh = build()
    by_name = {c["name"]: c for c in fresh["cases"]}
    for case in golden["cases"]:
        assert case["name"] in by_name, f"case {case['name']} no longer builds"
        assert case["expected"] == by_name[case["name"]]["expected"], (
            f"golden values for {case['name']} are stale" + REGOLD)


def test_the_two_copies_of_the_golden_values_agree(golden):
    from_json = json.loads((WEB / "tests" / "golden_values.json")
                           .read_text(encoding="utf-8"))
    assert from_json == golden, (
        "golden_values.json and js/golden.js disagree" + REGOLD)


def test_the_cases_move_every_control_the_page_exposes(golden):
    """A golden set where every case shares a gear ratio proves the bridge
    works for one number. The page has four controls; they all have to move,
    or the untested one is where the browser and the core diverge."""
    for field in ("gear_ratio", "wheel_radius_m", "total_mass_kg"):
        seen = {c["input"][field] for c in golden["cases"]}
        assert len(seen) >= 3, f"{field} barely varies across the cases: {seen}"
    assert {c["input"]["preset"] for c in golden["cases"]} >= {"m3508_c620",
                                                               "generic_dc"}
    assert any(c["input"]["datasheet_overrides"] for c in golden["cases"]), \
        "no case exercises a datasheet override"


def test_the_page_metrics_are_covered_not_just_the_raw_constants(golden):
    """Kt and tau are checked by every case. The numbers a reader actually acts
    on -- top speed, climb angle -- are computed by a formula duplicated in
    js/web-api.js, and a verification that skipped them would check the part
    that cannot drift while ignoring the part that can."""
    for case in golden["cases"]:
        page = case["expected"]["page"]
        assert set(page) == {"v_noload_axis_ms", "v_noload_diag_ms",
                             "push_stall_n", "climb_deg_at_stall",
                             "yaw_authority"}
        # The X layout makes the *axis* directions the fast ones, not the
        # diagonals: a wheel whose drive axis lies at 45 deg sees the whole
        # body speed when the body moves along that diagonal, and only
        # cos 45 of it when the body moves along +x. Same wheel speed limit,
        # so the body goes sqrt(2) faster along +x. Getting this backwards is
        # the standard way to over-report an omni's top speed, and it is why
        # the page shows both numbers instead of one labelled "max".
        # Not exact: the declared axis angles are rounded to 3 decimals.
        assert page["v_noload_axis_ms"] == pytest.approx(
            page["v_noload_diag_ms"] * 2 ** 0.5, rel=1e-3)


def test_the_shipped_chassis_still_has_two_wheels_with_no_yaw_authority(golden):
    """Not an approval -- a tripwire.

    config/robot/chassis.yaml declares drive axes for an X layout and then
    gives the wheels positions rotated 90 deg against them, so fl and fr
    contribute nothing to yaw and the robot turns on rl and rr alone. The
    jacobian stays invertible (condition number 3.3 against a limit of 1e6),
    so nothing in the simulator complains; the chassis page draws it and says
    so, which is how it was found.

    Fixing it is out of this brief's scope -- it moves every drivetrain-sweep
    and localisation result in the repo. When someone does fix it, this test
    fails, and that is the point: it is the note that cannot be lost.
    """
    for case in golden["cases"]:
        yaw = case["expected"]["page"]["yaw_authority"]
        assert sorted(yaw) == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-3), (
            "the chassis layout changed. If the wheel positions were fixed to "
            "match the declared drive axes, delete this test and re-run "
            "scripts/export_golden_values.py" + REGOLD)


def test_the_web_api_duplicates_the_climb_formula_faithfully(golden):
    """js/web-api.js re-implements max_climb_angle_rad because field/slope.py
    cannot be copied to the browser. Check the constant, at least, rather than
    trusting that two files written on the same afternoon stayed in step."""
    from omni_sim_core.field.slope import G
    text = (WEB / "js" / "web-api.js").read_text(encoding="utf-8")
    assert f"G = {G}" in text, "web-api.js and field/slope.py disagree about g"
    assert "0.5 * math.asin(ratio)" in text
    assert "math.pi / 4" in text, "the 45 deg saturation is missing"


def test_the_python_glue_survives_being_a_template_literal():
    """web-api.js carries its Python in a JS String.raw`...` literal, so a
    backtick in a docstring ends the literal and the whole page dies with
    "String.raw(...) is not a function" -- which no Python test can see, and
    which reST double-backticks produce on the first person's instinct.
    Cost me one round trip through a real browser; now it costs a test."""
    text = (WEB / "js" / "web-api.js").read_text(encoding="utf-8")
    body = text.split("String.raw`", 1)[1].rsplit("`;", 1)[0]
    assert "`" not in body, "a backtick inside the template literal ends it"
    assert "${" not in body, "${ inside a template literal interpolates"


def test_the_glue_is_valid_python():
    """It is a string as far as JS is concerned, so a syntax error in it
    surfaces only when Pyodide tries to run it, in someone's browser."""
    import ast

    text = (WEB / "js" / "web-api.js").read_text(encoding="utf-8")
    body = text.split("String.raw`", 1)[1].rsplit("`;", 1)[0]
    ast.parse(body)          # raises SyntaxError with a line number if not


def test_the_glue_and_the_exporter_agree_on_the_metrics():
    """Two copies of page_metrics exist on purpose (the browser cannot import
    mechanism/robot_config). They must at least still name the same outputs."""
    from export_golden_values import page_metrics as exporter
    import inspect

    text = (WEB / "js" / "web-api.js").read_text(encoding="utf-8")
    for key in ("v_noload_axis_ms", "v_noload_diag_ms", "push_stall_n",
                "climb_deg_at_stall", "yaw_authority"):
        assert f'"{key}"' in inspect.getsource(exporter), key
        assert f'"{key}"' in text, f"{key} missing from js/web-api.js"


# --------------------------------------------------------------------------- #
# the pages themselves
# --------------------------------------------------------------------------- #
PAGES = ["index.html"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_only_files_that_exist(page):
    """No build step means no bundler to catch a renamed script. A missing
    <script src> on file:// is a silent no-op until something calls into it."""
    html = (WEB / page).read_text(encoding="utf-8")
    base = (WEB / page).parent
    for src in re.findall(r'<(?:script src|link rel="stylesheet" href)="([^"]+)"', html):
        if src.startswith("http"):
            continue
        assert (base / src).resolve().exists(), f"{page} references missing {src}"


@pytest.mark.parametrize("page", PAGES)
def test_no_page_uses_an_es_module(page):
    """type="module" is blocked on file:// by the same opaque-origin rule that
    blocks fetch(). It fails without an error in the console that points at the
    cause, so it is worth a test rather than a comment."""
    html = (WEB / page).read_text(encoding="utf-8")
    # a real tag, not the string appearing in the page's own prose
    assert not re.search(r'<script[^>]*type="module"', html), (
        f"{page} uses an ES module, which cannot load from file://")


def test_the_shell_loads_every_panel_module():
    """The shell dispatches to OmniSimPanel / OmniChassisPanel /
    OmniVerifyPanel by name. A panel whose <script> is missing from the page
    fails only when someone clicks that entry in the sidebar -- not at load,
    and not in any Python test."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for src in ("js/vendor/roslib.min.js", "js/core-bundle.js", "js/golden.js",
                "js/web-api.js", "js/pyodide-bridge.js", "js/panel-sim.js",
                "js/panel-chassis.js", "js/panel-verify.js", "js/shell.js"):
        assert f'src="{src}"' in html, f"index.html does not load {src}"
    # shell.js must come last: it calls show() at the end of its own body,
    # which reaches straight into the panels.
    order = [html.index(f'src="js/{n}"') for n in
             ("panel-sim.js", "panel-chassis.js", "panel-verify.js", "shell.js")]
    assert order == sorted(order), "shell.js must load after the panels"


def test_the_moved_dashboard_kept_its_dom_contract():
    """js/panel-sim.js is a *move* of the dashboard's script, so index.html
    has to keep every id that script reaches for. Renaming one in the markup
    would break the live view while the chassis panel carried on looking
    perfect -- and only a click on 自動走行 would show it."""
    js = (WEB / "js" / "panel-sim.js").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    needed = set(re.findall(r'\$\("([A-Za-z][\w-]*)"\)', js))
    assert needed, "no $(\"id\") lookups found; did the extraction change shape?"
    missing = sorted(i for i in needed if f'id="{i}"' not in html)
    assert not missing, f"index.html is missing ids panel-sim.js uses: {missing}"


def test_the_sim_panel_was_not_rewritten():
    """The drawing code (walls, lethal overlay, robot footprint, particles)
    cost several debugging sessions to get right. Integrating the two pages
    was a styling job; if these functions stopped existing, something more
    than styling happened."""
    js = (WEB / "js" / "panel-sim.js").read_text(encoding="utf-8")
    for fn in ("drawWalls", "lethalOverlay", "drawRobot", "drawParticles",
               "drawMclGhost", "drawOdomGhost", "drawLevelEdge", "worldToCanvas"):
        assert f"function {fn}(" in js, f"{fn} is gone from the moved dashboard"


def test_the_service_worker_caches_what_the_page_actually_loads():
    """A stale LOCAL list is an offline cache that half works: the page loads
    and one panel is missing, which reads as a bug in the panel."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    sw = (WEB / "sw.js").read_text(encoding="utf-8")
    for src in re.findall(r'<script src="([^"h][^"]*)"', html):
        assert f'"./{src}"' in sw, f"sw.js does not cache {src}"
    for css in re.findall(r'<link rel="stylesheet" href="([^"h][^"]*)"', html):
        assert f'"./{css}"' in sw, f"sw.js does not cache {css}"
