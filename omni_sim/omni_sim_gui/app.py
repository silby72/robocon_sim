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
from .pages.actuator_page import ActuatorPage


def build_window(config_dir: Path, start_tab: str = "chassis") -> QtWidgets.QMainWindow:
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"omni_sim config editor  ·  Qt binding: {BINDING}")
    tabs = QtWidgets.QTabWidget()
    order = ["chassis", "actuators", "sensors"]
    tabs.addTab(ChassisPage(config_dir / "chassis.yaml"), "Chassis")
    tabs.addTab(ActuatorPage(config_dir), "Actuators")
    # placeholder tab for the sensors page (Phase D)
    ph = QtWidgets.QLabel("  Sensors page — coming next")
    ph.setAlignment(_align_center())
    tabs.addTab(ph, "Sensors")
    if start_tab in order:
        tabs.setCurrentIndex(order.index(start_tab))
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
    ap.add_argument("--tab", default="chassis",
                    choices=["chassis", "actuators", "sensors"],
                    help="which tab to open on")
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = build_window(Path(args.config_dir).resolve(), start_tab=args.tab)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
