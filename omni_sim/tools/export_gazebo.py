#!/usr/bin/env python3
"""CLI for omni_sim_core.export.gazebo_actuator (avoids the runpy -m warning).

    python tools/export_gazebo.py config/generated/plant_true.yaml \
        results/gazebo/drivetrain.yaml --chassis config/robot/chassis.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omni_sim_core" / "src"))

from omni_sim_core.export.gazebo_actuator import main  # noqa: E402

raise SystemExit(main())
