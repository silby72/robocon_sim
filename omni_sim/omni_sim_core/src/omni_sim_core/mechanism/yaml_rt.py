"""Comment-preserving YAML round-trip helpers (ruamel.yaml).

Input : a path to a YAML file. Output: a ``CommentedMap`` that remembers key
order and comments, so a load -> save with no edits is byte-for-byte identical.
Writes are atomic (temp file + replace). This is the single place the mechanism
layer touches YAML; the GUI's own ``io/yaml_io.py`` (Phase B) can build on top.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _represent_none(representer, _data):
    # Emit an explicit ``null`` (not an empty value) so authored ``null`` tokens
    # survive a round-trip unchanged.
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


def _yaml() -> YAML:
    y = YAML()  # round-trip mode by default (preserves comments + order)
    y.preserve_quotes = True
    y.width = 4096          # do not wrap long lines / lists
    y.indent(mapping=2, sequence=4, offset=2)
    y.representer.add_representer(type(None), _represent_none)
    return y


def rt_load(path: str | Path) -> Any:
    """Load a YAML file in round-trip mode (comments and order preserved)."""
    with open(path, encoding="utf-8") as f:
        return _yaml().load(f)


def rt_loads(text: str) -> Any:
    return _yaml().load(text)


def rt_dumps(data: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(data, buf)
    return buf.getvalue()


def rt_dump(data: Any, path: str | Path) -> None:
    """Atomically write ``data`` to ``path`` (temp file then os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _yaml().dump(data, f)
    os.replace(tmp, path)
