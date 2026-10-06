"""Compare vx tracking with the DOB on vs off, into one PNG.

Both runs command target_vx = 1.0 with the SAME PID gains; the only difference
is whether the disturbance observer is enabled. The nominal model is 10% light
and frictionless, so DOB-off leaves a steady-state error the DOB removes.

Explicit loops only (project rule).
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_column(path, column):
    times = []
    values = []
    input_file = open(path, "r", newline="")
    reader = csv.DictReader(input_file)
    for row in reader:
        times.append(float(row["t"]))
        values.append(float(row[column]))
    input_file.close()
    return times, values


def main():
    on_path = sys.argv[1]
    off_path = sys.argv[2]
    png_path = sys.argv[3]

    on_t, on_vx = load_column(on_path, "vx_body")
    off_t, off_vx = load_column(off_path, "vx_body")

    figure, axis = plt.subplots(1, 1, figsize=(10, 6))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                 label="target vx = 1.0")
    axis.plot(off_t, off_vx, color="tab:red", label="DOB off (PID only)")
    axis.plot(on_t, on_vx, color="tab:blue", label="DOB on (PID + DOB)")
    axis.set_title("Body-vx step tracking: nominal model is 10% light + "
                   "frictionless\nDOB reconstructs the mismatch")
    axis.set_xlabel("t [s]")
    axis.set_ylabel("vx_body [m/s]")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    figure.savefig(png_path, dpi=110)
    print("wrote", png_path)


if __name__ == "__main__":
    main()
