#!/usr/bin/env python3
"""One-shot, re-runnable dev environment setup for omni_sim.

Creates ``.venv``, installs ``omni_sim_core`` editable with its extras, wires
VS Code up to that interpreter, and installs the VS Code extensions this
project actually benefits from.

    python setup_env.py                 # do everything (safe to re-run)
    python setup_env.py --check         # report only, change nothing
    python setup_env.py --system        # install into the ACTIVE interpreter
                                        #   (use inside a sourced ROS 2 env --
                                        #    never build a venv on top of ROS)
    python setup_env.py --no-vscode     # skip editor setup
    python setup_env.py --verify        # also run the test suite + regen maps

Stdlib only, by necessity: this runs before anything is installed.

--------------------------------------------------------------------------
KEEP THIS UP TO DATE
Everything that can drift lives in the CONFIG block below. Python
dependencies are NOT listed here -- they are read from
``omni_sim_core/pyproject.toml`` so the two can never disagree. To add a
runtime dependency, edit pyproject.toml; to add an editor extension or a new
optional extra, edit the block below.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

# == CONFIG ================================================================= #

MIN_PYTHON = (3, 11)

# extras from pyproject's [project.optional-dependencies] installed by default
DEFAULT_EXTRAS = ["test", "viz"]

# VS Code extensions worth having for THIS repo, and why.
VSCODE_EXTENSIONS = {
    "ms-python.python":           "interpreter, test discovery, debugging",
    "ms-python.vscode-pylance":   "type checking across omni_sim_core",
    "ms-python.debugpy":          "step through run_sim.py / scenarios",
    "redhat.vscode-yaml":         "config/*.yaml, maps/field_2027.yaml",
    "tamasfe.even-better-toml":   "pyproject.toml",
    "mechatroner.rainbow-csv":    "results/*.csv from run_sim.py",
    "ms-iot.vscode-ros":          "omni_sim_ros (ROS 2 Jazzy) package",
}

# Modules that must import once setup is done -> (import name, what needs it).
# Import the SUBMODULE the code actually touches, not the top-level package:
# ``import scipy`` succeeds even when ``scipy.signal``'s compiled extension is
# unloadable, which is exactly the failure this check exists to catch.
SMOKE_IMPORTS = [
    ("numpy",             "core"),
    ("scipy.signal",      "control/dob.py -- cont2discrete"),
    ("yaml",              "config loading"),
    ("PIL.Image",         "maps/*.pgm"),
    ("matplotlib.pyplot", "viz extra"),
    ("pytest",            "test extra"),
]

GITIGNORE_LINES = [
    ".venv/",
    "__pycache__/",
    "*.py[cod]",
    "*.egg-info/",
    "results/",
    ".pytest_cache/",
]

# =========================================================================== #

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "omni_sim_core"
VENV = ROOT / ".venv"
REPO = ROOT.parent

BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    ("\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
    if sys.stdout.isatty() and os.name != "nt" or os.environ.get("WT_SESSION")
    else ("",) * 6
)


def step(msg: str) -> None:
    print(f"\n{BOLD}==> {msg}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}ok{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}warn{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET} {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  {DIM}$ {' '.join(str(c) for c in cmd)}{RESET}")
    return subprocess.run([str(c) for c in cmd], **kw)


# --------------------------------------------------------------------------- #
def venv_python(root: Path) -> Path:
    """Interpreter inside a venv, on either platform layout."""
    win = root / "Scripts" / "python.exe"
    return win if win.exists() or os.name == "nt" else root / "bin" / "python"


def read_extras() -> dict[str, list[str]]:
    with open(CORE / "pyproject.toml", "rb") as f:
        pp = tomllib.load(f)
    return pp.get("project", {}).get("optional-dependencies", {})


def check_python() -> bool:
    v = sys.version_info
    if v[:2] < MIN_PYTHON:
        fail(f"Python {v.major}.{v.minor} < required {'.'.join(map(str, MIN_PYTHON))}")
        return False
    ok(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    return True


def resolve_python(spec: str | None) -> Path | None:
    """Find the interpreter to build the venv from.

    ``spec`` is a path, or a bare version like ``3.12``. Versions are looked up
    through the ``py`` launcher (Windows) and then ``uv``, which will download
    the runtime if it is missing. ``None`` means "the one running this script".
    """
    if not spec:
        return Path(sys.executable)
    p = Path(spec)
    if p.exists():
        return p.resolve()

    if os.name == "nt" and shutil.which("py"):
        r = subprocess.run(["py", f"-{spec}", "-c", "import sys;print(sys.executable)"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())

    uv = shutil.which("uv")
    if uv:
        for args in (["python", "find", spec], ["python", "install", spec]):
            r = subprocess.run([uv, *args], capture_output=True, text=True)
            if args[1] == "find" and r.returncode == 0 and r.stdout.strip():
                return Path(r.stdout.strip().splitlines()[-1])
            if args[1] == "install" and r.returncode == 0:
                print(f"  {DIM}uv installed Python {spec}{RESET}")
        r = subprocess.run([uv, "python", "find", spec],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip().splitlines()[-1])

    fail(f"no Python {spec} found (tried a literal path, the 'py' launcher, uv)")
    return None


def venv_version(py: Path) -> str:
    r = subprocess.run([str(py), "-c",
                        "import sys;print('%d.%d.%d' % sys.version_info[:3])"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "?"


def ensure_venv(base: Path, check: bool, recreate: bool) -> Path:
    py = venv_python(VENV)
    want = venv_version(base)

    if py.exists():
        have = venv_version(py)
        if recreate:
            if check:
                warn(f"would delete and rebuild {VENV} (Python {have} -> {want})")
                return py
            print(f"  removing {VENV} (Python {have})")
            shutil.rmtree(VENV)
        elif have != want:
            warn(f"venv is Python {have}, requested {want} -- "
                 f"pass --recreate to rebuild it")
            ok(f"venv exists: {VENV} (Python {have})")
            return py
        else:
            ok(f"venv exists: {VENV} (Python {have})")
            return py

    if check:
        warn(f"venv missing: {VENV} (would build from Python {want})")
        return py

    print(f"  creating {VENV} from Python {want} ({base})")
    if base.resolve() == Path(sys.executable).resolve():
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(VENV)
    else:
        # a different interpreter has to build its own venv
        r = subprocess.run([str(base), "-m", "venv", str(VENV)])
        if r.returncode:
            fail(f"venv creation failed (exit {r.returncode})")
            return venv_python(VENV)
    ok(f"venv created: {VENV} (Python {want})")
    return venv_python(VENV)


def install(py: Path, extras: list[str], check: bool) -> bool:
    target = f"{CORE}[{','.join(extras)}]" if extras else str(CORE)
    if check:
        warn(f"would install -e {target}")
        return True
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    r = run([py, "-m", "pip", "install", "-e", target])
    if r.returncode:
        fail(f"pip install -e {target} failed (exit {r.returncode})")
        return False
    ok(f"installed -e omni_sim_core[{','.join(extras)}]")
    return True


def smoke(py: Path) -> bool:
    """Import every dependency in the target interpreter, report per-module."""
    probe = (
        "import importlib, sys, json\n"
        "out = {}\n"
        "for m in sys.argv[1:]:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        top = sys.modules[m.split('.')[0]]\n"
        "        out[m] = {'v': getattr(top, '__version__', 'ok')}\n"
        "    except Exception as e:\n"
        "        out[m] = {'err': f'{type(e).__name__}: {e}'}\n"
        "print(json.dumps(out))\n"
    )
    names = [m for m, _ in SMOKE_IMPORTS]
    # not via run(): the probe is multi-line, echoing it would bury the report
    r = subprocess.run([str(py), "-c", probe, *names],
                       capture_output=True, text=True)
    if r.returncode:
        fail("import probe crashed:\n" + (r.stderr or "").strip())
        return False

    found = json.loads(r.stdout)
    good, errors = True, []
    for mod, why in SMOKE_IMPORTS:
        res = found.get(mod, {})
        if "v" in res:
            ok(f"{mod:<18} {res['v']:<10} {DIM}{why}{RESET}")
        else:
            fail(f"{mod:<18} {'BROKEN':<10} {DIM}{why}{RESET}")
            print(f"       {DIM}{res.get('err', 'not installed')}{RESET}")
            errors.append(res.get("err", ""))
            good = False
    if not good:
        diagnose(errors)
    return good


def diagnose(errors: list[str]) -> None:
    """Turn the common, non-obvious import failures into an actionable note."""
    if os.name != "nt" or not any("DLL load failed" in e for e in errors):
        return
    state = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\CI\Policy") as k:
            state = winreg.QueryValueEx(k, "VerifiedAndReputablePolicyState")[0]
    except OSError:
        pass
    if state not in (1, 2):
        return
    print(f"\n  {YELLOW}Smart App Control is "
          f"{'enforced' if state == 1 else 'in evaluation mode'}{RESET} on this "
          f"machine, and it blocks freshly published, low-reputation binary\n"
          f"  wheels -- that is what 'DLL load failed' above means. The package "
          f"is fine; Windows refuses to load its .pyd.\n"
          f"  Wheels for the newest Python release are the usual casualty, "
          f"because they have had no time to build reputation.\n\n"
          f"  {BOLD}Fix, cheapest first:{RESET}\n"
          f"    1. build the venv on an older Python, whose wheels are already "
          f"trusted:\n"
          f"       {BOLD}python setup_env.py --python 3.12 --recreate{RESET}\n"
          f"    2. develop in WSL2 -- you need Linux for ROS 2 Jazzy anyway\n"
          f"    3. turn Smart App Control off in Windows Security. "
          f"{RED}This is irreversible{RESET}:\n"
          f"       re-enabling it requires reinstalling Windows. Your call, "
          f"not this script's.\n")


# --------------------------------------------------------------------------- #
def setup_vscode(py: Path, check: bool) -> None:
    d = ROOT / ".vscode"
    interp = str(py).replace("\\", "/")

    recs = {"recommendations": sorted(VSCODE_EXTENSIONS)}
    settings = {
        "python.defaultInterpreterPath": interp,
        "python.testing.pytestEnabled": True,
        "python.testing.pytestArgs": ["omni_sim_core/tests"],
        "python.analysis.extraPaths": ["omni_sim_core/src"],
        "files.associations": {"*.pgm": "plaintext"},
        "yaml.schemas": {},
    }

    if check:
        warn(f"would write {d}/extensions.json and settings.json")
    else:
        d.mkdir(exist_ok=True)
        # merge, never clobber: the user's own settings win
        sp = d / "settings.json"
        if sp.exists():
            try:
                settings = {**settings, **json.loads(sp.read_text(encoding="utf-8"))}
            except json.JSONDecodeError:
                warn("existing settings.json is not valid JSON -- leaving it alone")
                settings = None
        if settings is not None:
            sp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        (d / "extensions.json").write_text(
            json.dumps(recs, indent=2) + "\n", encoding="utf-8")
        ok(f"wrote {d.name}/extensions.json + settings.json")

    code = shutil.which("code") or shutil.which("code.cmd")
    if not code:
        warn("'code' CLI not on PATH -- VS Code will prompt for the recommended "
             "extensions when you next open this folder")
        return

    r = subprocess.run([code, "--list-extensions"], capture_output=True, text=True)
    have = {ln.strip().lower() for ln in r.stdout.splitlines() if ln.strip()}
    missing = [e for e in VSCODE_EXTENSIONS if e.lower() not in have]
    if not missing:
        ok(f"all {len(VSCODE_EXTENSIONS)} extensions already installed")
        return
    if check:
        for e in missing:
            warn(f"would install {e}  {DIM}({VSCODE_EXTENSIONS[e]}){RESET}")
        return
    for e in missing:
        r = run([code, "--install-extension", e, "--force"],
                capture_output=True, text=True)
        (ok if r.returncode == 0 else fail)(
            f"{e}  {DIM}{VSCODE_EXTENSIONS[e]}{RESET}")


def setup_gitignore(check: bool) -> None:
    gi = REPO / ".gitignore"
    have = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    missing = [ln for ln in GITIGNORE_LINES if ln not in have]
    if not missing:
        ok(".gitignore up to date")
        return
    if check:
        warn(f"would add to .gitignore: {', '.join(missing)}")
        return
    body = ("\n".join(have).rstrip() + "\n\n" if have else "")
    gi.write_text(body + "\n".join(missing) + "\n", encoding="utf-8")
    ok(f"{'updated' if have else 'created'} {gi} (+{len(missing)} rules)")


def verify(py: Path) -> bool:
    good = True
    r = run([py, "-m", "pytest", CORE / "tests", "-q"], cwd=ROOT)
    (ok if r.returncode == 0 else fail)("test suite")
    good &= r.returncode == 0
    r = run([py, ROOT / "maps" / "gen_field_2027.py"], cwd=ROOT)
    (ok if r.returncode == 0 else fail)("map generation")
    return good and r.returncode == 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what would change, touch nothing")
    ap.add_argument("--system", action="store_true",
                    help="install into the active interpreter, no venv "
                         "(this is what you want inside a sourced ROS 2 env)")
    ap.add_argument("--extras", default=",".join(DEFAULT_EXTRAS),
                    help=f"pyproject extras to install (default: "
                         f"{','.join(DEFAULT_EXTRAS)}; '' for none)")
    ap.add_argument("--python", metavar="VER|PATH",
                    help="interpreter to build the venv from, e.g. '3.12' or a "
                         "full path (default: the one running this script). "
                         "A bare version is resolved via the 'py' launcher, "
                         "then uv, which will download it if needed.")
    ap.add_argument("--recreate", action="store_true",
                    help="delete and rebuild the venv (needed to change its "
                         "Python version)")
    ap.add_argument("--no-vscode", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="after installing, run the tests and regenerate the maps")
    args = ap.parse_args()

    known = read_extras()
    extras = [e for e in args.extras.split(",") if e]
    unknown = [e for e in extras if e not in known]
    if unknown:
        fail(f"unknown extras {unknown}; pyproject.toml offers {sorted(known)}")
        return 2

    print(f"{BOLD}omni_sim setup{RESET}  {DIM}{ROOT}{RESET}")
    if args.check:
        print(f"{YELLOW}--check: nothing will be modified{RESET}")

    step("interpreter")
    if not check_python():
        return 2

    if args.system:
        if args.python:
            fail("--system and --python are mutually exclusive")
            return 2
        py = Path(sys.executable)
        warn("--system: installing into the active interpreter, no venv")
    else:
        base = resolve_python(args.python)
        if base is None:
            return 2
        if args.python:
            ok(f"base interpreter: Python {venv_version(base)} ({base})")
        step("virtual environment")
        py = ensure_venv(base, args.check, args.recreate)

    step(f"python packages  {DIM}(extras: {', '.join(extras) or 'none'}){RESET}")
    installed = install(py, extras, args.check) if (py.exists() or args.check) else False

    step("dependency check")
    healthy = smoke(py) if py.exists() else (warn("no interpreter yet"), False)[1]

    if not args.no_vscode:
        step("vs code")
        setup_vscode(py, args.check)

    step("git")
    setup_gitignore(args.check)

    if args.verify and not args.check and healthy:
        step("verify")
        healthy &= verify(py)

    step("next")
    if args.check:
        print("  re-run without --check to apply.")
        return 0
    act = (VENV / "Scripts" / "activate") if os.name == "nt" else (VENV / "bin" / "activate")
    if not args.system:
        print(f"  activate:  {BOLD}{act}{RESET}"
              f"   {DIM}(bash: source {act}){RESET}")
    print(f"  test:      python -m pytest omni_sim_core/tests -q")
    print(f"  run:       python run_sim.py --scenario config/scenarios/dob_step_load.yaml --ui")
    print(f"  maps:      python maps/gen_field_2027.py")
    if not (installed and healthy):
        print(f"\n{RED}setup incomplete -- see FAIL lines above{RESET}")
        return 1
    print(f"\n{GREEN}environment ready{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
