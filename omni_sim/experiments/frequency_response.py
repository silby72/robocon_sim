"""Numerically estimate the sensitivity S and complementary sensitivity T of the
motor speed loop from a chirp disturbance, and plot the Bode magnitude.

The chirp is injected as a torque disturbance; the transfer from disturbance to
the tracking error (proportional to S) is estimated by the ratio of cross- and
auto-spectral densities (Welch). Requires matplotlib.

    python experiments/frequency_response.py --out results/bode.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "omni_sim_core" / "src"))

from omni_sim_core.clock import ClockConfig                       # noqa: E402
from omni_sim_core.plant.motor import MotorParams                 # noqa: E402
from omni_sim_core.plant.nominal import nominal_from_true         # noqa: E402
from omni_sim_core.control.pid import PIDParams                   # noqa: E402
from omni_sim_core.control.dob import DOBParams                   # noqa: E402
from omni_sim_core.disturbance import DisturbanceSpec             # noqa: E402
from omni_sim_core.simulator import MotorControlSim               # noqa: E402


def estimate_sensitivity(dob_enabled: bool, duration: float = 20.0):
    true = MotorParams(n_motors=1)
    nominal = nominal_from_true(true, j_ratio=0.5, b_ratio=1.0)
    chirp = [DisturbanceSpec(target="motor_torque", index=0, waveform="chirp",
                             amplitude=0.1, freq_start_hz=0.1, freq_end_hz=200.0,
                             start_s=0.0, stop_s=duration)]
    clk = ClockConfig(1e-4, 1e-3, 2e-2)
    sim = MotorControlSim(
        true_params=true, nominal=nominal,
        pid_params=PIDParams(kp=0.05, ki=0.5),
        dob_params=DOBParams(enabled=dob_enabled, tau_q=0.005, order=1),
        clock_cfg=clk, seed=0, disturbances=chirp, duration_s=duration,
        setpoint_fn=lambda t: 0.0,
    )
    log = sim.run()
    fs = 1.0 / clk.dt_sim
    d = log["d_true"]
    err = log["omega_ref"] - log["omega_true"]  # response to the disturbance
    f, Pdd = signal.welch(d, fs, nperseg=8192)
    _, Pde = signal.csd(d, err, fs, nperseg=8192)
    S = np.abs(Pde / Pdd)
    return f, S


def main(argv: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bode.png")
    args = ap.parse_args(argv)

    f_off, S_off = estimate_sensitivity(False)
    f_on, S_on = estimate_sensitivity(True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f_off[1:], 20 * np.log10(S_off[1:] + 1e-12), label="DOB off")
    ax.semilogx(f_on[1:], 20 * np.log10(S_on[1:] + 1e-12), label="DOB on")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|S| [dB] (disturbance -> error)")
    ax.set_title("Sensitivity, disturbance rejection")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[freq] wrote {out}")


if __name__ == "__main__":
    main()
