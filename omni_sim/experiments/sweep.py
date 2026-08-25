"""Parallel parameter sweep over DOB / model-error settings.

Example: tau_q (10 values) x J_ratio (5 values) = 50 conditions in parallel.

    python experiments/sweep.py --tau-q 0.002 0.02 --n-tau 10 \
        --j-ratio 0.25 1.5 --n-j 5 --out results/sweep.csv

Each condition runs headless (no ROS) via MotorControlSim and reports a couple
of scalar metrics (steady error, disturbance-rejection drop).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# make the core importable when run from a source checkout
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.clock import ClockConfig                       # noqa: E402
from omni_sim_core.plant.motor import MotorParams                 # noqa: E402
from omni_sim_core.plant.nominal import nominal_from_true         # noqa: E402
from omni_sim_core.control.pid import PIDParams                   # noqa: E402
from omni_sim_core.control.dob import DOBParams                   # noqa: E402
from omni_sim_core.disturbance import DisturbanceSpec             # noqa: E402
from omni_sim_core.simulator import MotorControlSim               # noqa: E402


@dataclass
class Condition:
    tau_q: float
    j_ratio: float


def _run_one(cond: Condition) -> dict:
    true = MotorParams(n_motors=1)
    nominal = nominal_from_true(true, j_ratio=cond.j_ratio, b_ratio=1.0)
    load = [DisturbanceSpec(target="motor_torque", index=0, waveform="step",
                            amplitude=0.2, start_s=0.5)]
    sim = MotorControlSim(
        true_params=true, nominal=nominal,
        pid_params=PIDParams(kp=0.02, ki=0.0),
        dob_params=DOBParams(enabled=True, tau_q=cond.tau_q, order=1),
        clock_cfg=ClockConfig(1e-4, 1e-3, 2e-2),
        seed=0, disturbances=load, duration_s=1.5,
        setpoint_fn=lambda t: 50.0,
    )
    log = sim.run()
    t = log["t"]
    pre = np.mean(log["omega_true"][(t > 0.45) & (t < 0.5)])
    post = np.mean(log["omega_true"][t > 1.45])
    return {
        "tau_q": cond.tau_q,
        "j_ratio": cond.j_ratio,
        "load_drop": abs(post - pre),
        "steady_error": abs(50.0 - post),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DOB parameter sweep")
    ap.add_argument("--tau-q", type=float, nargs=2, default=[0.002, 0.02])
    ap.add_argument("--n-tau", type=int, default=10)
    ap.add_argument("--j-ratio", type=float, nargs=2, default=[0.25, 1.5])
    ap.add_argument("--n-j", type=int, default=5)
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--jobs", type=int, default=0, help="0 => all cores")
    args = ap.parse_args(argv)

    tau_qs = np.linspace(args.tau_q[0], args.tau_q[1], args.n_tau)
    j_ratios = np.linspace(args.j_ratio[0], args.j_ratio[1], args.n_j)
    conds = [Condition(tq, jr) for tq, jr in product(tau_qs, j_ratios)]

    jobs = args.jobs or None
    with Pool(processes=jobs) as pool:
        rows = pool.map(_run_one, conds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[sweep] {len(rows)} conditions -> {out}")


if __name__ == "__main__":
    main()
