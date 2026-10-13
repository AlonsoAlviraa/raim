"""R9: parameter-plateau stability and plateau-centre selection (replaces argmax in wf.py).

Neighbours of grid point k: points that differ in exactly ONE parameter by one step in that
parameter's ordered list of values (order of first appearance for categoricals)."""
from __future__ import annotations

import numpy as np


def _axes(grid: list[dict]) -> dict:
    axes = {}
    for p in grid:
        for k, v in p.items():
            axes.setdefault(k, [])
            if v not in axes[k]:
                axes[k].append(v)
    for k, vals in axes.items():
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            axes[k] = sorted(vals)
    return axes


def neighbors(grid: list[dict], k: int) -> list[int]:
    axes, p, out = _axes(grid), grid[k], []
    for j, q in enumerate(grid):
        if j == k or set(q) != set(p):
            continue
        diff = [a for a in p if q[a] != p[a]]
        if len(diff) == 1:
            a = diff[0]
            if abs(axes[a].index(q[a]) - axes[a].index(p[a])) == 1:
                out.append(j)
    return out


def stability(scores, grid: list[dict], k: int, keep: float = 0.7) -> float | None:
    nb = neighbors(grid, k)
    if not nb:
        return None
    s0 = scores[k]
    if not np.isfinite(s0) or s0 <= 0:
        return 0.0
    return float(np.mean([scores[j] >= keep * s0 for j in nb]))


def heatmap(scores, grid: list[dict]) -> list[dict]:
    return [{**p, "score": float(scores[k]), "stability": stability(scores, grid, k)}
            for k, p in enumerate(grid)]


def plateau_select(scores, grid: list[dict]) -> int:
    """Pick the centre of the best plateau: maximise the MEDIAN score over the closed
    neighbourhood (self + neighbours) instead of the single best point."""
    best, best_v = 0, -np.inf
    for k in range(len(grid)):
        hood = [scores[k]] + [scores[j] for j in neighbors(grid, k)]
        v = float(np.median(hood))
        if v > best_v or (v == best_v and scores[k] > scores[best]):
            best, best_v = k, v
    return best
