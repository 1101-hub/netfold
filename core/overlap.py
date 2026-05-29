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


def sat_intersect(va: np.ndarray, vb: np.ndarray, eps: float = _EPS) -> bool:
    """
    True only if triangles overlap with positive area.
    Edge-touching (min overlap depth ≈ 0) returns False.
    """
    min_overlap = np.inf
    for axis in np.vstack([_edge_normals(va), _edge_normals(vb)]):
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
    # fold-adjacent pairs — skip these
    adjacent: set[frozenset] = {
        frozenset((fe.tri_a, fe.tri_b)) for fe in nf.fold_edges
    }

    ids   = [t.id for t in nf.triangles]
    verts = [t.vertices.astype(float) for t in nf.triangles]
    n     = len(ids)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((ids[i], ids[j])) in adjacent:
                continue
            if sat_intersect(verts[i], verts[j], eps):
                pairs.append((ids[i], ids[j]))

    return OverlapResult(pairs=pairs)