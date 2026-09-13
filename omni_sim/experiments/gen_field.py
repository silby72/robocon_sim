#!/usr/bin/env python3
"""Slice ``config/field/robocon2027.yaml`` into map_server grids (依頼 §3).

One spec -> per-layer nav + localization grids (PGM + YAML), the exact format
``env/occupancy_grid.py`` reads and RViz2 displays.

    python experiments/gen_field.py                       # all layers, defaults
    python experiments/gen_field.py --set l1_barrier_h=300
    python experiments/gen_field.py --layer l1 --out maps/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.field import LayeredField          # noqa: E402


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k.strip()] = float(v)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path,
                    default=ROOT / "config" / "field" / "robocon2027.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "maps" / "field2027")
    ap.add_argument("--layer", action="append", help="repeatable; default = all")
    ap.add_argument("--set", action="append", metavar="NAME=MM",
                    help="override an unknown, e.g. --set l1_barrier_h=300")
    args = ap.parse_args(argv)

    field = LayeredField.from_yaml(args.spec, unknowns=_parse_set(args.set))
    paths = field.write(args.out, layers=args.layer)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
