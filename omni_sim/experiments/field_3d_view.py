#!/usr/bin/env python3
"""Render the 2027 field as a 3D colored overview.

Examples:
    python3 experiments/field_3d_view.py --save results/field_2027_3d.png
    MPLBACKEND=Agg python3 experiments/field_3d_view.py --save results/field_2027_3d.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _rect_to_box(rect):
    if isinstance(rect, dict):
        rect = rect["rect"]
    x0, y0, x1, y1 = rect
    return (x0, y0, x1 - x0, y1 - y0)


def _add_rect_prism(ax, rect, z0, dz, color, alpha=0.90, edgecolor="none"):
    x0, y0, dx, dy = _rect_to_box(rect)
    ax.bar3d(
        x0, y0, z0, dx, dy, dz,
        color=color,
        alpha=alpha,
        shade=True,
        edgecolor=edgecolor,
        linewidth=0.2,
        zsort="average",
    )


def _add_cylinder(ax, center, radius, z0, height, color, alpha=0.9):
    cx, cy = center
    x = np.linspace(cx - radius, cx + radius, 40)
    y = np.linspace(cy - radius, cy + radius, 40)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    Z = np.zeros_like(R)
    mask = R <= radius
    Z[mask] = z0 + height
    ax.plot_surface(X, Y, np.where(mask, z0 + height, z0), color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(X, Y, np.where(mask, z0 + height, z0), color=color, alpha=alpha, linewidth=0)


def build_3d_scene(spec):
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")

    colors = spec["colors"]
    color_map = {k: _rgb_to_hex(v) for k, v in colors.items()}

    field_w, field_h = spec["field"]["size"]
    ax.set_xlim(0, field_w)
    ax.set_ylim(0, field_h)
    ax.set_zlim(0, 1.2)
    ax.set_box_aspect((field_w, field_h, 1.0))

    # ground field colors
    ax.bar3d(0, 0, 0, field_w / 2.0, field_h, 0.03,
             color=color_map["ground_red"], alpha=0.9, edgecolor="none")
    ax.bar3d(field_w / 2.0, 0, 0, field_w / 2.0, field_h, 0.03,
             color=color_map["ground_blue"], alpha=0.9, edgecolor="none")

    # shared and storage zones
    rect = spec["zones"]["ground_shared"]
    _add_rect_prism(ax, rect, 0.03, 0.02, color_map["ground_shared"], alpha=0.8)

    for team in ("red", "blue"):
        rect = spec["zones"]["storage"][team]
        _add_rect_prism(ax, rect, 0.03, 0.02, color_map[f"start_{team}"], alpha=0.75)

    # L1 platform
    l1_rect = spec["platforms"]["l1"]["rect"]
    _add_rect_prism(ax, l1_rect, 0.60, 0.08, color_map["l1_red"], alpha=0.85)

    # L2 platform
    l2_rect = spec["platforms"]["l2"]["rect"]
    _add_rect_prism(ax, l2_rect, 0.90, 0.08, color_map["l2_area"], alpha=0.9)

    # ramps and stairs as elevated blocks
    for team in ("red", "blue"):
        ramp = spec["ramps"][team]["rect"]
        _add_rect_prism(ax, ramp, 0.03, 0.58, color_map[f"ramp_{team}"], alpha=0.8)
        stair = spec["stairs"][f"l1_{team}"]["rect"]
        _add_rect_prism(ax, stair, 0.03, 0.57, color_map[f"l1_{team}"], alpha=0.8)

    # pillars and fences
    mustika = spec["pillars"]["mustika"]
    cx, cy = mustika["center"]
    _add_cylinder(ax, (cx, cy), mustika["diameter"] / 2.0, 0.03, mustika["height"], color_map["pillar"], alpha=0.9)

    central = spec["pillars"]["central"]
    cx, cy = central["center"]
    _add_cylinder(ax, (cx, cy), central["diameter"] / 2.0, 0.90, central["height"], color_map["pillar"], alpha=0.9)

    for seg in spec["fences"]["center_divider_ground"]:
        x0, y0, x1, y1 = seg["seg"]
        ax.plot([x0, x1], [y0, y1], [0.0, 0.0], color=color_map["boundary"], linewidth=2.0)

    for seg in spec["fences"]["center_divider_l1"]:
        x0, y0, x1, y1 = seg["seg"]
        ax.plot([x0, x1], [y0, y1], [0.60, 0.60], color=color_map["boundary"], linewidth=2.0)

    # markers and labels
    ax.set_title("ABU Robocon 2027 Field (3D overview)", color="white", pad=20)
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.set_zlabel("z [m]", color="white")
    ax.tick_params(colors="white")
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((0.12, 0.12, 0.12, 1.0))
    ax.yaxis.pane.set_facecolor((0.12, 0.12, 0.12, 1.0))
    ax.zaxis.pane.set_facecolor((0.12, 0.12, 0.12, 1.0))
    ax.view_init(elev=26, azim=42)

    return fig, ax


def main():
    ap = argparse.ArgumentParser(description="Render the 2027 field as a 3D colored overview")
    ap.add_argument("--spec", default=str(ROOT / "maps" / "field_2027.yaml"), help="spec path")
    ap.add_argument("--save", default=str(ROOT / "results" / "field_2027_3d.png"), help="output image path")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    fig, _ = build_3d_scene(spec)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[field_3d_view] wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
