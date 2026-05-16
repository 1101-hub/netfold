"""
NetFold Cube Encoder
=====================
Encodes a unit cube as a NetFold.

Net layout (cross, like your drawing):
         [top]
  [left][front][right][back]
        [bottom]

Each face = 2 triangles.
  tri_A: vertices BL, BR, TR
  tri_B: vertices BL, TR, TL

Author: Amulya
"""

import numpy as np
from .structure import NetFold, Triangle2D, FoldEdge, StitchEdge, RootAnchor

FOLD_90  = np.pi / 2
FOLD_180 = np.pi


def _make_face(col: int, row: int, start_id: int):
    c, r = float(col), float(row)
    corners = np.array([
        [c,   r  ],
        [c+1, r  ],
        [c+1, r+1],
        [c,   r+1],
    ])
    tA = Triangle2D(id=start_id,   vertices=corners[[0, 1, 2]])
    tB = Triangle2D(id=start_id+1, vertices=corners[[0, 2, 3]])
    return tA, tB


def build_cube_net() -> NetFold:
    faces = {}
    tid = 0
    for name, (col, row) in [
        ('top',    (1, 0)),
        ('front',  (1, 1)),
        ('bottom', (1, 2)),
        ('back',   (1, 3)),
        ('left',   (0, 1)),
        ('right',  (2, 1)),
    ]:
        tA, tB = _make_face(col, row, tid)
        faces[name] = {'A': tA, 'B': tB}
        tid += 2

    triangles = []
    for f in faces.values():
        triangles.extend([f['A'], f['B']])

    # Within-face diagonal edges (180° flat)
    fold_edges = []
    for f in faces.values():
        fold_edges.append(FoldEdge(
            tri_a=f['A'].id, tri_b=f['B'].id,
            local_a=(0, 2), local_b=(0, 1),
            dihedral_angle=FOLD_180, fold_direction=1
        ))

    def fold(fA_tri, fA_local, fB_tri, fB_local):
        fold_edges.append(FoldEdge(
            tri_a=fA_tri.id, tri_b=fB_tri.id,
            local_a=fA_local, local_b=fB_local,
            dihedral_angle=FOLD_90, fold_direction=-1
        ))

    fold(faces['front']['B'], (1,2), faces['top']['A'],    (0,1))
    fold(faces['front']['A'], (0,1), faces['bottom']['B'], (1,2))
    fold(faces['front']['B'], (0,2), faces['left']['A'],   (1,2))
    fold(faces['front']['A'], (1,2), faces['right']['B'],  (0,2))
    fold(faces['bottom']['A'], (0,1), faces['back']['B'],  (1,2))

    # Correct stitch edges — computed from ground truth auto-detection
    stitch_edges = [
        StitchEdge(0,  11, (1,2), (1,2), FOLD_90),
        StitchEdge(1,  7,  (1,2), (1,2), FOLD_90),
        StitchEdge(1,  9,  (2,0), (1,2), FOLD_90),
        StitchEdge(4,  10, (1,2), (0,1), FOLD_90),
        StitchEdge(5,  6,  (1,2), (0,1), FOLD_90),
        StitchEdge(5,  8,  (2,0), (0,1), FOLD_90),
        StitchEdge(7,  10, (2,0), (1,2), FOLD_90),
    ]

    root_id = faces['front']['A'].id
    root_3d = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    root = RootAnchor(
        triangle_id=root_id,
        position_3d=root_3d,
        normal_3d=np.array([0.0, 0.0, -1.0])
    )

    nf = NetFold(
        name="unit_cube",
        triangles=triangles,
        fold_edges=fold_edges,
        stitch_edges=stitch_edges,
        root=root
    )
    nf.build_adjacency()
    return nf


def get_ground_truth_positions(nf: NetFold) -> dict:
    """
    Hardcoded correct 3D positions for every triangle.

    Cube vertices:
      A=(0,0,0) B=(1,0,0) C=(1,1,0) D=(0,1,0)
      E=(0,0,1) F=(1,0,1) G=(1,1,1) H=(0,1,1)
    """
    A = np.array([0.,0.,0.])
    B = np.array([1.,0.,0.])
    C = np.array([1.,1.,0.])
    D = np.array([0.,1.,0.])
    E = np.array([0.,0.,1.])
    F = np.array([1.,0.,1.])
    G = np.array([1.,1.,1.])
    H = np.array([0.,1.,1.])

    faces = {}
    tid = 0
    for name in ['top','front','bottom','back','left','right']:
        faces[name] = {'A_id': tid, 'B_id': tid+1}
        tid += 2

    placed = {
        faces['top']['A_id']:    np.array([D, C, G]),
        faces['top']['B_id']:    np.array([D, G, H]),
        faces['front']['A_id']:  np.array([A, B, C]),
        faces['front']['B_id']:  np.array([A, C, D]),
        faces['bottom']['A_id']: np.array([A, B, F]),
        faces['bottom']['B_id']: np.array([A, F, E]),
        faces['back']['A_id']:   np.array([F, E, H]),
        faces['back']['B_id']:   np.array([F, H, G]),
        faces['left']['A_id']:   np.array([E, A, D]),
        faces['left']['B_id']:   np.array([E, D, H]),
        faces['right']['A_id']:  np.array([B, F, G]),
        faces['right']['B_id']:  np.array([B, G, C]),
    }
    return placed