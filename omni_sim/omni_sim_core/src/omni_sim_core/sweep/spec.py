"""Declare a parameter space in YAML and expand it to conditions (依頼 §5.1).

    parameters:
      footprint_width: {range: [0.40, 1.00], step: 0.02}   # inclusive range
      l1_barrier_h:    {values: [0.10, 0.30]}              # explicit list
      planner:         {values: [astar, astar_shortcut]}

Cartesian product, in declaration order (stable => reproducible). ``range`` is
inclusive of both ends; the step count is rounded so float drift never drops or
duplicates the last point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import yaml


def _expand_one(name: str, spec: dict) -> list:
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        lo, hi = (float(v) for v in spec["range"])
        step = float(spec["step"])
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + i * step, 12) for i in range(n)]
    raise ValueError(f"parameter {name!r} needs 'values' or 'range'+'step'")


@dataclass
class SweepSpec:
    name: str
    parameters: dict[str, dict]
    metrics: list[str] = field(default_factory=list)
    parallel: int = 1
    seed: int = 0
    base_scenario: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepSpec":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {"name", "parameters", "metrics", "parallel", "seed",
                 "base_scenario"}
        return cls(
            name=cfg.get("name", "sweep"),
            parameters=cfg["parameters"],
            metrics=list(cfg.get("metrics", [])),
            parallel=int(cfg.get("parallel", 1)),
            seed=int(cfg.get("seed", 0)),
            base_scenario=cfg.get("base_scenario"),
            extra={k: v for k, v in cfg.items() if k not in known},
        )

    def conditions(self) -> list[dict]:
        """Every combination as an ordered dict of param -> value."""
        names = list(self.parameters)
        axes = [_expand_one(n, self.parameters[n]) for n in names]
        return [dict(zip(names, combo)) for combo in product(*axes)]
