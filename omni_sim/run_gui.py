#!/usr/bin/env python3
"""Launch the config editor GUI from the project root (no PYTHONPATH needed).

    python3 run_gui.py

Injects omni_sim_core/src and the repo root onto sys.path, then starts
omni_sim_gui.app. Requires a Qt binding (PySide6 preferred, PyQt5 fallback) and
ruamel.yaml -- install with `python setup_env.py --extras gui`.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))
sys.path.insert(0, str(ROOT))

from omni_sim_gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
