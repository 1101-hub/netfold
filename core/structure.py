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
class CutEdge:
    """
    An edge that was severed to prevent 2D net overlap.

    In the original mesh, this edge connects two adjacent triangles.
    In the 2D net, they are placed on separate islands.
    During 3D reconstruction, cut edges behave like stitch edges —
    the triangles must meet but are folded independently.
    """
    tri_a: int
    tri_b: int
    local_a: Tuple[int, int]
    local_b: Tuple[int, int]
    dihedral_angle: float
    fold_direction: int


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

    Optional fields for normalised meshes:
      scale     : float — bounding-box scale factor applied during encoding
      translate : (3,) array — translation applied during encoding
    To recover original coordinates: original = reconstructed * scale + translate
    """
    name: str
    triangles: List[Triangle2D]
    fold_edges: List[FoldEdge]
    stitch_edges: List[StitchEdge]
    root: RootAnchor

    # Multi-island support (empty = single island, backward compat)
    cut_edges: List = field(default_factory=list)
    islands: List[List[int]] = field(default_factory=list)  # [[tri_ids], ...]

    # Normalisation metadata (None = mesh was not normalised)
    scale: float = None
    translate: np.ndarray = None

    # Built from fold_edges — not stored, derived
    _adjacency: Dict = field(default_factory=dict, repr=False)

    def build_adjacency(self):
        """Build BFS traversal graph from fold edges."""
        self._adjacency = {t.id: [] for t in self.triangles}
        for fe in self.fold_edges:
            self._adjacency[fe.tri_a].append((fe.tri_b, fe))
            self._adjacency[fe.tri_b].append((fe.tri_a, fe))

    @property
    def n_islands(self) -> int:
        return max(len(self.islands), 1)

    def calculate_weld_length(self) -> float:
        """
        Calculates the total linear distance of all cut seams (stitch edges) in 3D.
        This metric acts as a 'Manufacturing Cost / Structural Penalty' for optimization.
        """
        total_length = 0.0
        
        # We need to map tri_id -> 3D vertices to compute edge lengths.
        # But NetFold itself doesn't store the 3D vertices directly except root.
        # Actually, reconstruct() returns the 3D vertices. 
        # A simpler way: we can compute it if reconstruct() is passed in, 
        # or we just note that it should be run after reconstruct.
        # Wait, the length of the 3D edge is identical to the length of the 2D edge! (Isometry).
        for se in self.stitch_edges:
            tri_a = self.triangles[se.tri_a]
            p1 = tri_a.vertices[se.local_a[0]]
            p2 = tri_a.vertices[se.local_a[1]]
            
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            total_length += (dx**2 + dy**2)**0.5
            
        return total_length

    def summary(self):
        print("="*52)
        print(f"  NetFold: {self.name}")
        print("="*52)
        print(f"  Triangles    : {len(self.triangles)}")
        print(f"  Fold edges   : {len(self.fold_edges)}")
        print(f"  Stitch edges : {len(self.stitch_edges)}")
        print(f"  Cut edges    : {len(self.cut_edges)}")
        print(f"  Islands      : {self.n_islands}")
        
        # Also print weld length!
        weld_len = self.calculate_weld_length()
        print(f"  Weld length  : {weld_len:.3f}")
        
        print(f"  Root triangle: {self.root.triangle_id}")
        if self.scale is not None:
            print(f"  Scale        : {self.scale:.6f}")
        if self.translate is not None:
            print(f"  Translate    : [{self.translate[0]:.4f}, {self.translate[1]:.4f}, {self.translate[2]:.4f}]")
        print("="*52)