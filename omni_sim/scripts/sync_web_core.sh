#!/usr/bin/env bash
# Copy the browser page's slice of omni_sim_core. See sync_web_core.py for
# what is copied and why the list is explicit rather than a glob.
#
#     scripts/sync_web_core.sh
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$REPO/scripts/sync_web_core.py" "$@"
