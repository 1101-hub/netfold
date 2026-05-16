"""
NetFold Format — Core Data Structures
======================================
A 3D surface encoded as:
  - 2D net of triangles (flat, unfolded)
  - Fold angle at every interior edge (float, radians)
  - Stitch list for closing edges
  - One root triangle as 3D anchor

Author: Amulya
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class Triangle2D:
    """A triangle in the 2D net."""
    id: int
    vertices: np.ndarray          # shape (3, 2) — 2D positions

    def centroid(self):
        return self.vertices.mean(axis=0)


@dataclass
class FoldEdge:
    """
    An interior edge shared by two triangles.
    Stores the dihedral angle to fold along during reconstruction.

    dihedral_angle: float in radians
        pi     = flat (180°) — no fold, within-face diagonal
        pi/2   = 90° fold — adjacent faces of a cube
        Other values for car panel geometry
    fold_direction: +1 or -1
        Determines which way the fold goes (mountain vs valley)
    """
    tri_a: int
    tri_b: int
    local_a: Tuple[int, int]      # vertex indices in tri_a forming shared edge
    local_b: Tuple[int, int]      # vertex indices in tri_b forming shared edge
    dihedral_angle: float
    fold_direction: int


@dataclass
class StitchEdge:
    """
    An edge pair NOT connected in the 2D net but that must
    meet in 3D — like the last fold that closes a box.
    These are verified after reconstruction.
    """
    tri_a: int
    tri_b: int
    local_a: Tuple[int, int]
    local_b: Tuple[int, int]
    dihedral_angle: float


@dataclass
class RootAnchor:
    """
    One triangle fixed in 3D space.
    All other triangles are placed relative to this via BFS.
    """
    triangle_id: int
    position_3d: np.ndarray       # shape (3, 3) — 3D vertex positions
    normal_3d: np.ndarray         # shape (3,) — outward normal


@dataclass
class NetFold:
    """
    The complete NetFold encoding of a 3D surface.

    This is the format. Everything needed to reconstruct
    the 3D shape is contained here. Nothing else required.
    """
    name: str
    triangles: List[Triangle2D]
    fold_edges: List[FoldEdge]
    stitch_edges: List[StitchEdge]
    root: RootAnchor

    # Built from fold_edges — not stored, derived
    _adjacency: Dict = field(default_factory=dict, repr=False)

    def build_adjacency(self):
        """Build BFS traversal graph from fold edges."""
        self._adjacency = {t.id: [] for t in self.triangles}
        for fe in self.fold_edges:
            self._adjacency[fe.tri_a].append((fe.tri_b, fe))
            self._adjacency[fe.tri_b].append((fe.tri_a, fe))

    def summary(self):
        print(f"\n{'='*52}")
        print(f"  NetFold: {self.name}")
        print(f"{'='*52}")
        print(f"  Triangles    : {len(self.triangles)}")
        print(f"  Fold edges   : {len(self.fold_edges)}")
        print(f"  Stitch edges : {len(self.stitch_edges)}")
        print(f"  Root triangle: {self.root.triangle_id}")
        print(f"{'='*52}\n")