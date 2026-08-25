"""ANSI panel dashboard, matching the serial2can (serial_bridge/ros2can) look.

Same palette, panel width, spinner and clear-and-redraw approach as
``ros2can/console_ui.py`` and ``serial_bridge/graphical_ui.hpp``, generalised so
the simulator can feed it arbitrary sections.
"""
from __future__ import annotations

import sys
import time
import unicodedata
from dataclasses import dataclass, field

# -- palette (identical codes to serial2can) --------------------------------
_RESET = "\033[0m"
_FG_MUTED = "\033[38;5;245m"
_FG_TITLE = "\033[38;5;45m"
_FG_ACCENT = "\033[38;5;81m"
_FG_GOOD = "\033[38;5;48m"
_FG_GOOD_MID = "\033[38;5;120m"
_FG_WARN = "\033[38;5;214m"
_FG_WARN_ORANGE = "\033[38;5;208m"
_FG_BAD = "\033[38;5;203m"
_FG_TEXT = "\033[38;5;252m"


class C:
    """Named colours for callers building coloured content lines."""

    reset = _RESET
    muted = _FG_MUTED
    title = _FG_TITLE
    accent = _FG_ACCENT
    good = _FG_GOOD
    good_mid = _FG_GOOD_MID
    warn = _FG_WARN
    warn_orange = _FG_WARN_ORANGE
    bad = _FG_BAD
    text = _FG_TEXT


_PANEL_WIDTH = 100
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _spinner(tick: int) -> str:
    return _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]


def _char_width(c: str) -> int:
    return 2 if unicodedata.east_asian_width(c) in ("F", "W") else 1


def _visible_and_fit(text: str, width: int) -> str:
    """Fit to ``width`` display columns, ignoring ANSI escapes in the count and
    padding on the right. Colour codes pass through untouched."""
    if width <= 0:
        return ""
    out: list[str] = []
    visible = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\033" and i + 1 < n and text[i + 1] == "[":
            j = i + 2
            while j < n and not text[j].isalpha():
                j += 1
            if j < n:
                j += 1
            out.append(text[i:j])
            i = j
            continue
        w = _char_width(ch)
        if visible + w > width:
            break
        out.append(ch)
        visible += w
        i += 1
    pad = width - visible
    if pad > 0:
        out.append(" " * pad)
    return "".join(out)


def _border(color: str = _FG_MUTED) -> str:
    return f"{color}+{'-' * (_PANEL_WIDTH - 2)}+{_RESET}"


def _line(content: str, content_color: str = _FG_TEXT,
          border_color: str = _FG_MUTED) -> str:
    body = _visible_and_fit(f"{content_color}{content}", _PANEL_WIDTH - 4)
    return f"{border_color}| {body}{border_color} |{_RESET}"


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    title_color: str = _FG_ACCENT


def _force_utf8_stdout() -> None:
    """Make stdout able to carry the panel glyphs.

    The dashboard draws with box characters, a braille spinner and U+00B7. On a
    Japanese Windows install stdout defaults to cp932, which cannot encode any
    of them, and the first ``render`` dies with UnicodeEncodeError. Nothing
    here is lossy for an already-UTF-8 stream, and ``errors="replace"`` keeps a
    hostile console from taking the simulation down with it.
    """
    enc = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    if enc in ("utf8", "utf8mb4"):
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass  # not a reconfigurable stream; the glyphs degrade, the run lives


class ConsoleDashboard:
    """Clear-and-redraw terminal dashboard.

    Usage::

        dash = ConsoleDashboard("omni_sim live")
        dash.start()
        dash.render(status_line, [Section("SIM", [...]), ...])
        ...
        dash.stop()

    ``render`` is cheap but still writes the whole screen; throttle calls with
    ``should_render()`` if you drive it from a fast loop.
    """

    def __init__(self, title: str, min_period_s: float = 1.0 / 30.0) -> None:
        self.title = title
        self._tick = 0
        self._started = False
        self._min_period = min_period_s
        self._last_render = 0.0

    def start(self) -> None:
        _force_utf8_stdout()
        sys.stdout.write("\033[?25l")  # hide cursor
        sys.stdout.flush()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        sys.stdout.write("\033[?25h")  # restore cursor
        sys.stdout.flush()
        self._started = False

    def should_render(self) -> bool:
        now = time.monotonic()
        if now - self._last_render >= self._min_period:
            self._last_render = now
            return True
        return False

    def render(self, status: str, sections: list[Section]) -> None:
        self._tick += 1
        lines = [
            _border(_FG_TITLE),
            _line(f"{self.title}  ◉", _FG_TITLE, _FG_TITLE),
            _line(f"{_FG_GOOD}{_spinner(self._tick)} {status}{_RESET}",
                  _FG_MUTED, _FG_TITLE),
            _border(_FG_TITLE),
        ]
        for sec in sections:
            lines.append(_line(sec.title, sec.title_color))
            if not sec.lines:
                lines.append(_line("(none)", _FG_MUTED))
            for content in sec.lines:
                lines.append(_line(content))
            lines.append(_border())
        lines.append(_line("Ctrl+C to stop", _FG_MUTED))

        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


# -- small formatting helpers callers can reuse -----------------------------
def bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    """A simple horizontal gauge like ``[####----]`` scaled to [lo, hi]."""
    if hi <= lo:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def health_color(value: float, warn: float, bad: float,
                 higher_is_worse: bool = True) -> str:
    """Pick good/warn/bad by threshold."""
    if higher_is_worse:
        if value >= bad:
            return _FG_BAD
        if value >= warn:
            return _FG_WARN
        return _FG_GOOD
    else:
        if value <= bad:
            return _FG_BAD
        if value <= warn:
            return _FG_WARN
        return _FG_GOOD
