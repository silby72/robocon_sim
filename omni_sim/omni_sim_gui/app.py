"""omni_sim_gui entry point.

Opens the robot-description editor. For this first agile slice it shows page 1
(chassis). More pages (actuators, sensors) become tabs later.

    python run_gui.py [--config-dir config/robot]

The GUI only ever writes YAML under the chosen config dir; it never runs the
simulator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .qt import QtWidgets, BINDING
from .pages.chassis_page import ChassisPage


def build_window(config_dir: Path) -> QtWidgets.QMainWindow:
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"omni_sim config editor  ·  Qt binding: {BINDING}")
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(ChassisPage(config_dir / "chassis.yaml"), "Chassis")
    # placeholder tabs for the pages still to come (Phase C / D)
    for name in ("Actuators", "Sensors"):
        ph = QtWidgets.QLabel(f"  {name} page — coming next")
        ph.setAlignment(_align_center())
        tabs.addTab(ph, name)
    win.setCentralWidget(tabs)
    win.resize(1100, 720)
    return win


def _align_center():
    from .qt import QtCore
    return QtCore.Qt.AlignCenter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="omni_sim config editor")
    ap.add_argument("--config-dir", default="config/robot",
                    help="directory holding chassis.yaml etc.")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = build_window(Path(args.config_dir).resolve())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
