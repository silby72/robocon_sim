#!/usr/bin/env bash
# Build omni_sim_ros into the colcon workspace and make it runnable.
#
# Two things colcon gets wrong for this project, both of which have cost a
# debugging session before:
#
#   1. setuptools installs the console script to install/<pkg>/bin/, but
#      `ros2 run` / launch look in install/<pkg>/lib/<pkg>/. Without the
#      symlink the node is simply "not found".
#   2. the generated entry point gets `#!/usr/bin/python3`, which cannot see
#      omni_sim_core (it is installed editable in omni_sim/.venv). The failure
#      is a ModuleNotFoundError from deep inside importlib, which does not
#      look like a packaging problem at all.
#
# Both have to be redone after *every* build, so do not run colcon by hand.
#
#     scripts/build_ros.sh
set -eo pipefail   # not -u: ROS setup.bash reads unset AMENT_* variables

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${OMNI_SIM_WS:-$(dirname "$REPO")/ros2_ws}"
VENV_PY="$REPO/.venv/bin/python"

[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY -- run setup_env.py first" >&2; exit 1; }

source /opt/ros/jazzy/setup.bash
cd "$WS"
colcon build --packages-select omni_sim_ros --symlink-install

BIN="$WS/install/omni_sim_ros/bin/sim_node"
LIBDIR="$WS/install/omni_sim_ros/lib/omni_sim_ros"
sed -i "1s|.*|#!$VENV_PY|" "$BIN"
mkdir -p "$LIBDIR"
ln -sfn "$BIN" "$LIBDIR/sim_node"

echo "ok: $LIBDIR/sim_node -> $BIN  (interpreter $VENV_PY)"
