"""ABU Robocon 2027 (Solo, Indonesia) field visual layout for the 2D live viewer.

Dimensions and RGB colors are taken directly from the rulebook (Sections 1-2
for sizes, Section 14 for colors). Positions the rulebook does not pin down
exactly (e.g. where inside "each team's Ground Area" the Storage Area sits)
follow the same coordinate frame and the same [R]/[F]/[A] provenance
convention as ``config/field/robocon2027.yaml``:

    [R] a number stated in the rulebook text  -> trust it
    [F] scaled off Figure 2 / Figure 3        -> a reasonable estimate
    [A] no number in the rulebook at all       -> assumption

This module is purely a *visual* aid for ``ui/viewer.py`` -- collision/LiDAR
geometry lives in ``field/spec.py`` + the generated ``maps/field_2027_*``
grids, and the two are not required to be pixel-identical.

Frame: origin (0,0) = SW corner, +x east (red -> blue), +y north
(start zones -> storage/Mustika), matching ``config/field/robocon2027.yaml``.
All positions are millimetres here; ``draw_field_2027`` converts to metres.
"""
from __future__ import annotations

FIELD_SIZE_MM = 11000.0                # [R]
CENTER_X_MM = 5500.0                   # [R] centre divider

# Geometry also consumed by leveled_field_2027.py for collision/territory/
# ramp-gating, so the picture and the physics agree on where things are.
# Same [R]/[F]/[A] provenance as _zones() below.
L1_RECT_MM = (2500, 2500, CENTER_X_MM, 8500)           # [R] size; red half, mirror for blue
L2_RECT_MM = (4000, 4000, 7000, 7000)                  # [R] size, [F] centred on field
GROUND_SHARED_RECT_MM = (4900, 400, 6100, 1600)        # [R] size, [F] south-centre position
MUSTIKA_SHARED_RECT_MM = (5000, 7500, 6000, 8500)      # [R] size/position (matches the L1 notch)
RAMP_RECT_MM = (1500, 3000, 2500, 6500)                # [R] 3500 length, [F] pos/width; red, mirror for blue
L1_SHARED_STRIP_MM = (4700, 2500, 6300, 8500)          # [A] "boundary between Red/Blue L1"; 1600 wide
STAIRS_GATE_RECT_MM = (4700, 3700, 6300, 4300)         # [A] L1<->L2 crossing, straddles L2's south edge

# The functional crossing gates (leveled_field_2027.resolve_level) are more
# generous than the rectangles drawn on screen, but *only* in the direction
# actually being crossed (x here, since both gates connect levels split
# along x... no -- the Ramp crosses in x, the Stairs in y; each extends only
# its own crossing axis below). While the centre is inside a gate, both the
# old and new level are valid for every footprint corner (see
# resolve_level), so this margin has to comfortably clear how far a
# corner can lead or trail the centre: up to a 0.9 m square's half-diagonal,
# ~0.636 m. 800 mm covers that with room to spare, without widening the
# gate's cross-axis (the Ramp's y-span, the Stairs' x-span), which stays at
# the real dimension -- see leveled_field_2027.resolve_level's docstring for
# why over-widening the cross-axis caused its own, worse failure mode.
_GATE_MARGIN_MM = 700

# The Ramp gate must *contain the whole ramp*, not just its far end. The field
# spec carves the ramp FREE on the L1 grid too (connectors, see
# field/spec.ConnectorSpec), which is what gives the planner a connected
# corridor -- but it also means resolve_level's "prefer the highest level whose
# grid is free here" fallback reads any centre standing on the ramp as being on
# L1, and then judges the footprint against L1 alone, where everything off the
# 1.0 m ramp strip is a fall. A robot at the ramp's foot, safely on flat
# ground, was being declared in collision and jammed there.
#
# So the gate is grown past the carve on every side, not for the usual
# crossing-margin reason but so that no point that the carve can promote to L1
# ever falls outside the gate. That is also the physically honest model: the
# whole ramp is a straddle zone where the robot is on neither floor, and the
# cross-axis growth is 100 mm (covering the carve's 25 mm overshoot and
# _free_nearby's 1-cell = 20 mm tolerance), not the 700 mm that the docstring
# in leveled_field_2027.resolve_level warns about.
_RAMP_CARVE_SLACK_MM = 100
RAMP_GATE_RECT_MM = (RAMP_RECT_MM[0] - _RAMP_CARVE_SLACK_MM,
                     RAMP_RECT_MM[1] - _RAMP_CARVE_SLACK_MM,
                     RAMP_RECT_MM[2] + _GATE_MARGIN_MM,
                     RAMP_RECT_MM[3] + _RAMP_CARVE_SLACK_MM)
STAIRS_GATE_MARGIN_MM = (STAIRS_GATE_RECT_MM[0], STAIRS_GATE_RECT_MM[1] - _GATE_MARGIN_MM,
                         STAIRS_GATE_RECT_MM[2], STAIRS_GATE_RECT_MM[3] + _GATE_MARGIN_MM)


def _rgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255, g / 255, b / 255)


# straight out of rulebook Section 14 (Colors and Materials Specification)
_C = {
    "ground_red": _rgb(240, 210, 210), "ground_blue": _rgb(170, 210, 230),
    "shared": _rgb(245, 240, 200),
    "retry_red": _rgb(223, 34, 34), "retry_blue": _rgb(50, 0, 255),
    "l1_red": _rgb(235, 180, 160), "l1_blue": _rgb(150, 215, 220),
    "l2": _rgb(190, 190, 185),
    "building_spot": _rgb(40, 100, 50),
    "transfer_red": _rgb(245, 170, 60), "transfer_blue": _rgb(60, 170, 245),
    "ramp_red": _rgb(200, 150, 140), "ramp_blue": _rgb(130, 180, 200),
    "pillar": _rgb(100, 62, 0),
    "boundary": _rgb(100, 62, 0),
}


def _mirror(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reflect a (x0, y0, x1, y1) rect across the centre divider."""
    x0, y0, x1, y1 = rect
    return (FIELD_SIZE_MM - x1, y0, FIELD_SIZE_MM - x0, y1)


def _zones() -> list[tuple[str, tuple, tuple, float, str | None]]:
    """(kind, geometry_mm, rgb_color, zorder, label) for every static zone."""
    z: list[tuple[str, tuple, tuple, float, str | None]] = []

    # --- Ground Area halves (lowest layer) --------------------------------
    z.append(("rect", (0, 0, CENTER_X_MM, FIELD_SIZE_MM), _C["ground_red"], 0, None))
    z.append(("rect", (CENTER_X_MM, 0, FIELD_SIZE_MM, FIELD_SIZE_MM), _C["ground_blue"], 0, None))

    # --- Ground Shared Area: 1200x1200 sky-block grid, [F] south-centre ---
    z.append(("rect", GROUND_SHARED_RECT_MM, _C["shared"], 1, "Ground\nShared"))
    z += _sky_block_grid(GROUND_SHARED_RECT_MM)

    # --- 1000x1000 shared area around the Mustika Pillar -------------------
    z.append(("rect", MUSTIKA_SHARED_RECT_MM, _C["shared"], 1, None))          # [R] size/pos (matches L1 notch)

    # --- Storage Area 1000x2000, [F] corner near the north edge -----------
    storage_red = (300, 9000, 1300, 11000)                                    # [R] size, [F] pos
    z.append(("rect", storage_red, _C["ground_red"], 1, "Storage"))
    z.append(("rect", _mirror(storage_red), _C["ground_blue"], 1, None))

    # --- Start Zone / Ground Retry Zone: two 700x700 boxes, [F] SW corner --
    start_a = (300, 300, 1000, 1000)                                          # [R] size, [F] pos
    start_b = (1100, 300, 1800, 1000)
    for box in (start_a, start_b):
        z.append(("rect", box, _C["retry_red"], 2, None))
        z.append(("rect", _mirror(box), _C["retry_blue"], 2, None))
    z.append(("text", ((650 + 1450) / 2, 150), None, 2.5, "Start / Retry"))

    # --- Level 1 platform, split red/blue, notched for the Mustika ---------
    z.append(("rect", L1_RECT_MM, _C["l1_red"], 3, None))
    z.append(("rect", _mirror(L1_RECT_MM), _C["l1_blue"], 3, None))
    z.append(("rect", MUSTIKA_SHARED_RECT_MM, _C["shared"], 4, None))          # re-punch the Mustika notch
    z.append(("text", (4000, 3200), None, 4.4, "L1"))
    z.append(("text", (7000, 3200), None, 4.4, "L1"))

    # --- L1 Shared Area: the contested strip on the Red/Blue L1 boundary ---
    z.append(("rect", L1_SHARED_STRIP_MM, _C["shared"], 3.5, None))            # [A] width; outlined below

    # --- L1 Retry Zone 700x700, [F] north part of each team's L1 half ------
    l1_retry_red = (3000, 6700, 3700, 7400)                                   # [R] size, [F] pos
    z.append(("rect", l1_retry_red, _C["retry_red"], 4, None))
    z.append(("rect", _mirror(l1_retry_red), _C["retry_blue"], 4, None))

    # --- Ramp (3500 mm run) and Transfer Area (1000x1000), [F] west/east ---
    z.append(("rect", RAMP_RECT_MM, _C["ramp_red"], 2, "Ramp"))
    z.append(("rect", _mirror(RAMP_RECT_MM), _C["ramp_blue"], 2, None))

    transfer_red = (2000, 3000, 3000, 4000)                                   # [R] size, [F] pos (straddles L1 edge)
    z.append(("rect", transfer_red, _C["transfer_red"], 5, "Transfer"))
    z.append(("rect", _mirror(transfer_red), _C["transfer_blue"], 5, None))

    # --- Level 2 platform (fully shared) + Central Pillar -------------------
    z.append(("rect", L2_RECT_MM, _C["l2"], 6, None))
    z.append(("text", (5500, 6300), None, 6.4, "L2"))

    # --- L1<->L2 stairs gate: only place a robot may cross between them ----
    z.append(("rect", STAIRS_GATE_RECT_MM, _C["l1_red"], 6.2, "Stairs"))       # [A] position

    # --- Building Spots (500x500), [F] positions -----------------------------
    for cx, cy in ((3500, 3500), (7500, 3500), (5500, 3500)):
        z.append(("rect", (cx - 250, cy - 250, cx + 250, cy + 250),
                  _C["building_spot"], 5, None))
    z.append(("rect", (5250, 4250, 5750, 4750), _C["building_spot"], 7, None))  # L2 spot

    # --- Pillars --------------------------------------------------------
    z.append(("circle", (CENTER_X_MM, 8000, 270), _C["pillar"], 4.5, None))    # Mustika Pillar (ground)
    z.append(("circle", (CENTER_X_MM, CENTER_X_MM, 270), _C["pillar"], 7.5, None))  # Central Pillar (L2)

    # --- Centre divider fence + field boundary --------------------------
    z.append(("line", (CENTER_X_MM, 0, CENTER_X_MM, 8500), _C["boundary"], 8, None))  # [F] gaps omitted
    z.append(("outline", (0, 0, FIELD_SIZE_MM, FIELD_SIZE_MM), _C["boundary"], 9, None))

    return z


def _sky_block_grid(area: tuple[float, float, float, float]):
    """12 Sky Blocks on a 5x5 grid, alternating colour, centre empty (§4.1.4/Fig.4)."""
    x0, y0, x1, y1 = area
    cell = (x1 - x0) / 5.0
    out = []
    for row in range(5):
        for col in range(5):
            if row == 2 and col == 2:
                continue  # exact centre spot stays empty
            color = _C["retry_red"] if (row + col) % 2 == 0 else _C["retry_blue"]
            cx0 = x0 + col * cell + cell * 0.15
            cy0 = y0 + row * cell + cell * 0.15
            cx1 = x0 + (col + 1) * cell - cell * 0.15
            cy1 = y0 + (row + 1) * cell - cell * 0.15
            out.append(("rect", (cx0, cy0, cx1, cy1), color, 1.5, None))
    return out


def draw_field_2027(ax) -> None:
    """Draw the ABU Robocon 2027 field onto a matplotlib Axes (metres, top-down)."""
    from matplotlib.patches import Rectangle, Circle

    for kind, geom, color, zorder, label in _zones():
        if kind == "rect":
            x0, y0, x1, y1 = (v / 1000.0 for v in geom)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                                   edgecolor="#00000030", lw=0.4, zorder=zorder))
            if label:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center",
                        va="center", fontsize=6.5, color="#202020",
                        zorder=zorder + 0.4)
        elif kind == "circle":
            cx, cy, d = geom
            ax.add_patch(Circle((cx / 1000, cy / 1000), d / 2000.0,
                                facecolor=color, edgecolor="#00000060", lw=0.8,
                                zorder=zorder))
        elif kind == "line":
            x0, y0, x1, y1 = (v / 1000.0 for v in geom)
            ax.plot([x0, x1], [y0, y1], color=color, lw=3, zorder=zorder,
                    solid_capstyle="butt")
        elif kind == "outline":
            x0, y0, x1, y1 = (v / 1000.0 for v in geom)
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color,
                    lw=3, zorder=zorder)
        elif kind == "text":
            x, y = geom
            ax.text(x / 1000.0, y / 1000.0, label, ha="center", va="center",
                    fontsize=7, color="#404040", zorder=zorder)
