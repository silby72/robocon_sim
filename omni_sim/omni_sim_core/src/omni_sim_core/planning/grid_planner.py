"""8-connected A* over a :class:`CostField` (DECISIONS.md §2.1-2.3, §2.12).

Determinism is a hard requirement (invariant 6). The heap stores
``(f, h, counter, node)``: the ``h`` tie-break both improves efficiency (prefer
nodes closer to the goal at equal ``f``) and pins the expansion order, and the
monotone ``counter`` breaks any remaining ties so results are bit-identical
across runs (P1 acceptance #8).

Heuristic is the octile distance (§2.2), admissible for 8-connected moves and
tighter than Euclidean, so fewer nodes get expanded -- and ``expanded_nodes`` is
itself reported as an evaluation metric (§2.2, used by the sweep engine).

start / goal inside the inflated region is the *common* case (§2.3): we BFS to
the nearest free cell, shift there if it is within ``max_shift`` and record a
warning; otherwise we return ``PlanResult.failed`` -- never an exception.
"""
from __future__ import annotations

import heapq
import time
from collections import deque

import numpy as np

from ..env.occupancy_grid import OccupancyGrid
from .cost_field import CostField
from .types import Path, PlanConfig, PlanResult

SQRT2 = np.sqrt(2.0)

# 8-neighbour offsets (drow, dcol, step_len_in_cells, is_diagonal)
_NEIGHBORS = [
    (-1, 0, 1.0, False), (1, 0, 1.0, False),
    (0, -1, 1.0, False), (0, 1, 1.0, False),
    (-1, -1, SQRT2, True), (-1, 1, SQRT2, True),
    (1, -1, SQRT2, True), (1, 1, SQRT2, True),
]


class GridPlanner:
    """Reusable planner bound to one cost field (the map is static, §2.11)."""

    def __init__(self, cost_field: CostField) -> None:
        self.cf = cost_field
        self.grid = cost_field.grid

    # -- helpers ----------------------------------------------------------
    def _nearest_free_cell(self, row: int, col: int) -> tuple[int, int] | None:
        """BFS outward from (row, col) to the closest non-lethal cell (§2.3)."""
        cf = self.cf
        if not cf.is_lethal_cell(row, col):
            return (row, col)
        max_cells = int(np.ceil(self.cf.cfg.max_shift / cf.resolution))
        seen = np.zeros((cf.height, cf.width), dtype=bool)
        q: deque[tuple[int, int, int]] = deque([(row, col, 0)])
        seen[row, col] = True
        while q:
            r, c, d = q.popleft()
            if d > max_cells:
                continue
            if not cf.is_lethal_cell(r, c):
                return (r, c)
            for dr, dc, _, _ in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cf.height and 0 <= nc < cf.width and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc, d + 1))
        return None

    def _octile(self, r0: int, c0: int, r1: int, c1: int) -> float:
        dr = abs(r0 - r1)
        dc = abs(c0 - c1)
        lo, hi = (dr, dc) if dr < dc else (dc, dr)
        return (SQRT2 * lo + (hi - lo)) * self.cf.resolution

    # -- planning ---------------------------------------------------------
    def plan(self, start: tuple[float, float],
             goal: tuple[float, float]) -> PlanResult:
        t0 = time.perf_counter()
        cf = self.cf
        warnings: list[str] = []

        s_row, s_col = self.grid.world_to_grid(*start)
        g_row, g_col = self.grid.world_to_grid(*goal)

        for name, (r, c) in (("start", (s_row, s_col)), ("goal", (g_row, g_col))):
            if not (0 <= r < cf.height and 0 <= c < cf.width):
                return PlanResult.failed(
                    warnings=[f"{name} {(r, c)} is outside the map"],
                    elapsed_s=time.perf_counter() - t0)

        s_cell = self._nearest_free_cell(s_row, s_col)
        if s_cell is None:
            return PlanResult.failed(
                warnings=[f"start is inside inflation, no free cell within "
                          f"max_shift={cf.cfg.max_shift} m"],
                elapsed_s=time.perf_counter() - t0)
        if s_cell != (s_row, s_col):
            warnings.append(f"start shifted out of inflation to cell {s_cell}")

        g_cell = self._nearest_free_cell(g_row, g_col)
        if g_cell is None:
            return PlanResult.failed(
                warnings=[f"goal is inside inflation, no free cell within "
                          f"max_shift={cf.cfg.max_shift} m"],
                elapsed_s=time.perf_counter() - t0)
        if g_cell != (g_row, g_col):
            warnings.append(f"goal shifted out of inflation to cell {g_cell}")

        expanded, cells = self._astar(s_cell, g_cell)
        if cells is None:
            return PlanResult.failed(expanded_nodes=expanded, warnings=warnings,
                                     elapsed_s=time.perf_counter() - t0)

        pts = np.array([self.grid.grid_to_world(r, c) for r, c in cells],
                       dtype=float)
        return PlanResult(path=Path(pts), success=True, expanded_nodes=expanded,
                          warnings=warnings, elapsed_s=time.perf_counter() - t0)

    def _astar(self, start_cell, goal_cell):
        cf = self.cf
        H, W = cf.height, cf.width
        lethal = cf.lethal
        soft = cf.soft_cost
        res = cf.resolution
        allow_cut = cf.cfg.allow_diagonal_corner_cut

        g_score = np.full(H * W, np.inf)
        came_from = np.full(H * W, -1, dtype=np.int64)
        closed = np.zeros(H * W, dtype=bool)

        sr, sc = start_cell
        gr, gc = goal_cell
        s_idx = sr * W + sc
        g_idx = gr * W + gc
        g_score[s_idx] = 0.0

        counter = 0
        h0 = self._octile(sr, sc, gr, gc)
        heap: list[tuple[float, float, int, int]] = [(h0, h0, counter, s_idx)]
        expanded = 0

        while heap:
            f, h, _, idx = heapq.heappop(heap)
            if closed[idx]:
                continue
            closed[idx] = True
            expanded += 1
            if idx == g_idx:
                return expanded, self._reconstruct(came_from, s_idx, g_idx, W)

            r, c = divmod(idx, W)
            g_here = g_score[idx]
            for dr, dc, step, is_diag in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if lethal[nr, nc]:
                    continue
                if is_diag and not allow_cut:
                    # forbid squeezing diagonally between two lethal cells
                    if lethal[r, nc] or lethal[nr, c]:
                        continue
                n_idx = nr * W + nc
                if closed[n_idx]:
                    continue
                # edge cost = geometric length * (1 + soft cost of the target)
                step_cost = step * res * (1.0 + soft[nr, nc])
                tentative = g_here + step_cost
                if tentative < g_score[n_idx]:
                    g_score[n_idx] = tentative
                    came_from[n_idx] = idx
                    counter += 1
                    nh = self._octile(nr, nc, gr, gc)
                    heapq.heappush(heap, (tentative + nh, nh, counter, n_idx))

        return expanded, None

    @staticmethod
    def _reconstruct(came_from, s_idx, g_idx, W):
        cells = []
        idx = g_idx
        while idx != -1:
            cells.append(divmod(idx, W))
            if idx == s_idx:
                break
            idx = came_from[idx]
        cells.reverse()
        return cells


def plan(grid: OccupancyGrid, start: tuple[float, float],
         goal: tuple[float, float], cfg: PlanConfig | None = None) -> PlanResult:
    """Pure convenience wrapper: ``plan(grid, start, goal) -> PlanResult`` (§2.11)."""
    cfg = cfg or PlanConfig()
    return GridPlanner(CostField(grid, cfg)).plan(start, goal)
