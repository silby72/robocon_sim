"""Offline plotting of a scenario CSV (the default visualization path).

    python experiments/plot_results.py results/dob_step_load.csv --out results/dob.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(float(v))
    return {h: np.asarray(v) for h, v in cols.items()}


def main(argv: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    data = load_csv(args.csv)
    t = data["t"]
    has_dist = "d_hat" in data and "d_true" in data

    n = 2 if has_dist else 1
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax = axes[0]
    if "omega_ref" in data:
        ax.plot(t, data["omega_ref"], "k--", label="omega_ref")
    if "omega_true" in data:
        ax.plot(t, data["omega_true"], label="omega_true")
    ax.set_ylabel("omega [rad/s]")
    ax.grid(alpha=0.3)
    ax.legend()

    if has_dist:
        ax = axes[1]
        ax.plot(t, data["d_true"], "k--", label="d_true")
        ax.plot(t, data["d_hat"], label="d_hat (DOB estimate)")
        ax.set_ylabel("torque [N*m]")
        ax.set_xlabel("t [s]")
        ax.grid(alpha=0.3)
        ax.legend()
    else:
        axes[0].set_xlabel("t [s]")

    out = args.out or (Path(args.csv).with_suffix(".png"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
