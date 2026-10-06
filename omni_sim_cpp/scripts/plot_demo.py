"""Plot a demo CSV recorded by demo_run.py into a single PNG overview.

Explicit loops only (project rule). Headless: uses the Agg backend.
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    columns = {}
    input_file = open(path, "r", newline="")
    reader = csv.DictReader(input_file)
    for name in reader.fieldnames:
        columns[name] = []
    for row in reader:
        for name in reader.fieldnames:
            columns[name].append(float(row[name]))
    input_file.close()
    return columns


def main():
    csv_path = sys.argv[1]
    png_path = sys.argv[2]
    data = load_csv(csv_path)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    figure.suptitle("omni_sim_cpp demo run: forward -> spin -> strafe -> brake",
                    fontsize=14)

    # (0,0) XY trajectory (encoder dead-reckoned pose).
    trajectory = axes[0][0]
    trajectory.plot(data["odom_x"], data["odom_y"], color="tab:blue")
    trajectory.scatter([data["odom_x"][0]], [data["odom_y"][0]],
                       color="green", label="start", zorder=5)
    trajectory.scatter([data["odom_x"][-1]], [data["odom_y"][-1]],
                       color="red", label="end", zorder=5)
    trajectory.set_title("odometry path (x-y, world frame)")
    trajectory.set_xlabel("x [m]")
    trajectory.set_ylabel("y [m]")
    trajectory.axis("equal")
    trajectory.grid(True)
    trajectory.legend()

    # (0,1) body-frame velocities.
    velocity = axes[0][1]
    velocity.plot(data["t"], data["vx_body"], label="vx_body", color="tab:blue")
    velocity.plot(data["t"], data["vy_body"], label="vy_body", color="tab:orange")
    velocity.set_title("body-frame velocity (from encoder odom)")
    velocity.set_xlabel("t [s]")
    velocity.set_ylabel("[m/s]")
    velocity.grid(True)
    velocity.legend()

    # (1,0) yaw rate: odom vs noisy IMU gyro.
    yaw = axes[1][0]
    yaw.plot(data["t"], data["omega"], label="omega (odom)", color="tab:blue")
    yaw.plot(data["t"], data["imu_gyro_z"], label="gyro_z (IMU, noisy)",
             color="tab:red", alpha=0.6, linewidth=0.8)
    yaw.set_title("yaw rate: odometry vs noisy IMU gyro")
    yaw.set_xlabel("t [s]")
    yaw.set_ylabel("[rad/s]")
    yaw.grid(True)
    yaw.legend()

    # (1,1) noisy IMU body acceleration.
    accel = axes[1][1]
    accel.plot(data["t"], data["imu_accel_x"], label="accel_x (IMU)",
               color="tab:green", alpha=0.7, linewidth=0.8)
    accel.plot(data["t"], data["imu_accel_y"], label="accel_y (IMU)",
               color="tab:purple", alpha=0.7, linewidth=0.8)
    accel.set_title("IMU body acceleration (bias + white noise)")
    accel.set_xlabel("t [s]")
    accel.set_ylabel("[m/s^2]")
    accel.grid(True)
    accel.legend()

    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(png_path, dpi=110)
    print("wrote", png_path)


if __name__ == "__main__":
    main()
