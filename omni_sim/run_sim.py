#!/usr/bin/env python3
"""Convenience launcher so you can run the simulator from the project root
without setting PYTHONPATH:

    python3 run_sim.py --scenario config/scenarios/dob_step_load.yaml --ui

It just injects omni_sim_core/src onto sys.path and calls the real runner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "omni_sim_core" / "src"))

from omni_sim_core.run import main  # noqa: E402

if __name__ == "__main__":
    main()
