"""Sweep results as tidy rows (依頼 §5.1): one row per condition."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepResults:
    rows: list[dict]

    def ok(self) -> list[dict]:
        return [r for r in self.rows if r.get("_status") == "ok"]

    def failures(self) -> list[dict]:
        return [r for r in self.rows if r.get("_status") != "ok"]

    def mean_elapsed(self) -> float:
        el = [r.get("_elapsed_s", 0.0) for r in self.rows]
        return sum(el) / len(el) if el else 0.0

    def columns(self) -> list[str]:
        cols: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        return cols

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = self.columns()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow({c: r.get(c, "") for c in cols})
        return path

    def report(self) -> str:
        n = len(self.rows)
        fails = self.failures()
        lines = [f"sweep: {n} conditions, {n - len(fails)} ok, {len(fails)} failed",
                 f"mean time/condition: {self.mean_elapsed() * 1000:.1f} ms"]
        if fails:
            lines.append("failed conditions:")
            for r in fails:
                params = {k: v for k, v in r.items() if not k.startswith("_")}
                lines.append(f"  #{r.get('_index')} {params} -> {r.get('_error')}")
        return "\n".join(lines)
