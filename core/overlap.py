"""
overlap.py — 2D net overlap detection
Separating Axis Theorem on Triangle2D pairs.
Skips fold-adjacent pairs (they share an edge by construction).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .structure import NetFold

_EPS = 1e-9


# ── SAT core ──────────────────────────────────────────────────────────────────

def _edge_normals(v: np.ndarray) -> np.ndarray:
    """
    3 edge-perpendicular axes for a triangle.
    v: (3,2) → returns (3,2) normalized.
    """
    edges = np.roll(v, -1, axis=0) - v          # (3,2) edge vectors
    perp  = np.stack([-edges[:, 1], edges[:, 0]], axis=1)  # rotate 90°
    norms = np.linalg.norm(perp, axis=1, keepdims=True)
    return perp / np.where(norms > _EPS, norms, 1.0)


def _project(v: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    dots = v @ axis
    return float(dots.min()), float(dots.max())


def sat_intersect(
    va: np.ndarray,
    vb: np.ndarray,
    eps: float = _EPS,
    axes_a: np.ndarray | None = None,
    axes_b: np.ndarray | None = None,
) -> bool:
    """
    True only if triangles overlap with positive area.
    Edge-touching (min overlap depth ≈ 0) returns False.
    """
    if axes_a is None:
        axes_a = _edge_normals(va)
    if axes_b is None:
        axes_b = _edge_normals(vb)

    min_overlap = np.inf
    for axis in np.vstack([axes_a, axes_b]):
        lo_a, hi_a = _project(va, axis)
        lo_b, hi_b = _project(vb, axis)
        depth = min(hi_a, hi_b) - max(lo_a, lo_b)
        if depth < -eps:
            return False          # clear separation
        min_overlap = min(min_overlap, depth)
    return min_overlap > eps      # positive area overlap only

# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class OverlapResult:
    pairs: List[Tuple[int, int]]   # triangle ID pairs (not list indices)

    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def clean(self) -> bool:
        return not self.pairs

    def report(self):
        if self.clean:
            print("Overlap: CLEAN")
        else:
            print(f"Overlap: {self.count} pair(s)")
            for a, b in self.pairs:
                print(f"  tri {a} ↔ tri {b}")


# ── Main entry ────────────────────────────────────────────────────────────────

def find_overlaps(nf: NetFold, eps: float = _EPS) -> OverlapResult:
    """
    Check all non-adjacent triangle pairs in 2D net for overlap.

    Skips:  pairs sharing a fold edge (adjacent in BFS tree)
    Checks: stitch-edge pairs + all unrelated pairs

    Returns OverlapResult with triangle ID pairs that overlap.
    """
    # fold-adjacent pairs — skip these. Using sorted tuple is much faster than frozenset
    adjacent = {
        (min(fe.tri_a, fe.tri_b), max(fe.tri_a, fe.tri_b)) for fe in nf.fold_edges
    }

    ids   = [t.id for t in nf.triangles]
    verts = [t.vertices.astype(float) for t in nf.triangles]
    n     = len(ids)

    # Precompute edge normals
    normals = [_edge_normals(v) for v in verts]

    # Precompute axis-aligned bounding boxes (AABBs) for broad-phase collision filtering
    aabbs = []
    for v in verts:
        aabbs.append((v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))

    pairs = []
    for i in range(n):
        id_i = ids[i]
        vert_i = verts[i]
        norm_i = normals[i]
        min_x_i, max_x_i, min_y_i, max_y_i = aabbs[i]

        for j in range(i + 1, n):
            id_j = ids[j]
            if (min(id_i, id_j), max(id_i, id_j)) in adjacent:
                continue

            # Broad phase: Axis-Aligned Bounding Box (AABB) overlap check
            min_x_j, max_x_j, min_y_j, max_y_j = aabbs[j]
            if (max_x_i < min_x_j - eps or max_x_j < min_x_i - eps or
                max_y_i < min_y_j - eps or max_y_j < min_y_i - eps):
                continue

            # Narrow phase: Separating Axis Theorem (SAT)
            if sat_intersect(vert_i, verts[j], eps, norm_i, normals[j]):
                pairs.append((id_i, id_j))

    return OverlapResult(pairs=pairs)