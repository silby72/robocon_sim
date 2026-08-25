"""Headless scenario runner.

    python -m omni_sim_core.run --scenario config/scenarios/dob_step_load.yaml

Runs without ROS 2, writes a CSV of the requested signals, and exits.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .config import Scenario, load_scenario
from .simulator import MotorControlSim


def _step_setpoint(value: float):
    return lambda t: value


def run_scenario(scenario: Scenario, ui: bool = False,
                 rtf: float = 1.0) -> dict[str, np.ndarray]:
    true_params = scenario.resolve_true_params()
    nominal = scenario.resolve_nominal()
    ctrl = scenario.controller

    sim = MotorControlSim(
        true_params=true_params,
        nominal=nominal,
        pid_params=ctrl.pid,
        dob_params=ctrl.dob,
        clock_cfg=scenario.clock,
        seed=scenario.seed,
        disturbances=scenario.disturbance,
        duration_s=scenario.duration_s,
        setpoint_fn=_step_setpoint(ctrl.setpoint_omega_rad_s),
    )
    if not ui:
        return sim.run()

    from .ui import ConsoleDashboard
    from .ui.console import Section, C, bar, health_color

    dash = ConsoleDashboard(f"omni_sim  ·  {scenario.name}")
    dash.start()
    wall_start = time.monotonic()

    def observer(progress: float, st: dict) -> None:
        # real-time pacing so the run is actually watchable (rtf<=0 => fastest)
        if rtf > 0:
            target = wall_start + st["t"] / rtf
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        if not dash.should_render():
            return
        err = st["omega_ref"] - st["omega_true"]
        derr = st["d_true"] - st["d_hat"]
        ec = health_color(abs(err), warn=0.5, bad=2.0)
        dc = health_color(abs(derr), warn=0.02, bad=0.1)
        dob = (f"{C.good}ON {C.reset} tau_q={st['tau_q']*1e3:.1f} ms"
               if st["dob_enabled"] else f"{C.warn}OFF{C.reset}")
        sim_sec = Section("SIMULATION", [
            f"time   {st['t']:7.3f} / {scenario.duration_s:.2f} s   "
            f"{C.accent}{bar(progress, 0, 1)}{C.reset} {progress*100:4.0f}%",
            f"seed   {scenario.seed}    dt_sim {scenario.clock.dt_sim*1e3:.2f} ms   "
            f"dt_motor {scenario.clock.dt_motor*1e3:.2f} ms",
        ])
        ctl_sec = Section("MOTOR / DOB", [
            f"omega  {C.text}{st['omega_true']:8.3f}{C.reset}  ref "
            f"{st['omega_ref']:7.2f}   err {ec}{err:+7.3f}{C.reset} rad/s",
            f"torque tau_pid {st['tau_pid']:+7.3f}   tau_cmd {st['tau_cmd']:+7.3f} N·m",
            f"dist   d_true {st['d_true']:+7.3f}   d_hat {st['d_hat']:+7.3f}   "
            f"err {dc}{derr:+7.4f}{C.reset} N·m",
            f"DOB    {dob}",
        ])
        rtf_txt = f"rtf {rtf:.2f}" if rtf > 0 else "rtf max"
        dash.render(f"running  ·  {rtf_txt}", [sim_sec, ctl_sec])

    try:
        log = sim.run(observer=observer)
    finally:
        dash.stop()
    return log


def write_csv(log: dict[str, np.ndarray], out_path: str | Path,
              signals: list[str] | None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t"] + [s for s in (signals or list(log.keys())) if s != "t" and s in log]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        n = len(log["t"])
        for i in range(n):
            w.writerow([f"{log[c][i]:.9g}" for c in cols])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Headless omni_sim scenario runner")
    ap.add_argument("--scenario", required=True, help="path to scenario YAML")
    ap.add_argument("--output", default=None, help="override CSV output path")
    ap.add_argument("--ui", action="store_true",
                    help="live serial2can-style console dashboard while running")
    ap.add_argument("--rtf", type=float, default=1.0,
                    help="real-time factor for --ui (1.0=real time, 0=fastest)")
    args = ap.parse_args(argv)

    scenario = load_scenario(args.scenario)
    log = run_scenario(scenario, ui=args.ui, rtf=args.rtf)
    out = args.output or scenario._path(scenario.logging.output)
    write_csv(log, out, scenario.logging.signals)
    print(f"[omni_sim] scenario '{scenario.name}' done -> {out}")


if __name__ == "__main__":
    main()
