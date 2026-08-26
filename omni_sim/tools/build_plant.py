#!/usr/bin/env python3
"""CLI: build config/generated/plant_*.yaml from config/robot/*.

    python tools/build_plant.py            # build from the repo root
    python tools/build_plant.py --base .   # explicit base directory

Reads the robot description + motor presets and writes the generated plant and
jacobian files. Never overwrites the hand-written config/plant_*.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.mechanism.build import build  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build generated plant YAML files")
    ap.add_argument("--base", default=str(ROOT),
                    help="repo base dir containing config/ (default: repo root)")
    args = ap.parse_args(argv)

    result = build(args.base)
    print("[build_plant] wrote:")
    print(f"  true    -> {result.plant_true}")
    print(f"  nominal -> {result.plant_nominal}")
    print(f"  jacobian-> {result.jacobian}")


if __name__ == "__main__":
    main()
