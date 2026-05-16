"""
NetFold Reconstructor
=====================
Reconstruction via trilateration — places each free vertex
by solving distance constraints directly. No rotation ambiguity.

Author: Amulya
"""

import numpy as np
from collections import deque
from .structure import NetFold


def place_free_vertex(p0: np.ndarray, p1: np.ndarray,
                      d0: float, d1: float,
                      reference_normal: np.ndarray,
                      fold_direction: int,
                      fold_angle: float) -> np.ndarray:
    """
    Place a point that is distance d0 from p0 and d1 from p1,
    on the correct side determined by fold_direction.

    Uses the fold angle to determine which of the two possible
    positions (above/below the edge) is correct.
    """
    edge = p1 - p0
    edge_len = np.linalg.norm(edge)
    edge_unit = edge / edge_len

    # Foot of perpendicular from free vertex onto edge
    t = (d0**2 - d1**2 + edge_len**2) / (2 * edge_len**2)
    foot = p0 + t * edge

    # Height above edge
    h_sq = d0**2 - (t * edge_len)**2
    h = np.sqrt(max(h_sq, 0.0))

    # Two perpendicular directions to edge
    perp = np.cross(edge_unit, reference_normal)
    perp_len = np.linalg.norm(perp)
    if perp_len < 1e-10:
        # Edge parallel to normal — use fallback
        perp = np.array([1., 0., 0.])
        if abs(np.dot(perp, edge_unit)) > 0.9:
            perp = np.array([0., 1., 0.])
        perp = perp - np.dot(perp, edge_unit) * edge_unit
        perp /= np.linalg.norm(perp)
    else:
        perp /= perp_len

    # fold_direction determines which side
    return foot + fold_direction * h * perp


def reconstruct(nf: NetFold) -> dict:
    """
    Reconstruct 3D positions for all triangles via BFS.
    """
    placed = {}
    root_id = nf.root.triangle_id
    placed[root_id] = nf.root.position_3d.copy()

    queue = deque([root_id])
    visited = {root_id}

    while queue:
        current_id = queue.popleft()
        current_3d = placed[current_id]

        # Compute current face normal
        v0 = current_3d[1] - current_3d[0]
        v1 = current_3d[2] - current_3d[0]
        face_normal = np.cross(v0, v1)
        fn_len = np.linalg.norm(face_normal)
        if fn_len > 1e-12:
            face_normal /= fn_len
        else:
            face_normal = np.array([0., 0., 1.])

        for neighbour_id, fe in nf._adjacency[current_id]:
            if neighbour_id in visited:
                continue
            visited.add(neighbour_id)

            # Which local indices are current vs neighbour
            if fe.tri_a == current_id:
                local_current   = fe.local_a
                local_neighbour = fe.local_b
                direction       = fe.fold_direction
            else:
                local_current   = fe.local_b
                local_neighbour = fe.local_a
                direction       = -fe.fold_direction

            # Shared edge endpoints in 3D
            p0_3d = current_3d[local_current[0]]
            p1_3d = current_3d[local_current[1]]

            # Free vertex index in neighbour
            shared = list(local_neighbour)
            free   = [i for i in range(3) if i not in shared][0]

            # Distances from 2D (preserved by isometric folding)
            tri_2d   = nf.triangles[neighbour_id].vertices
            free_2d  = tri_2d[free]
            sh0_2d   = tri_2d[shared[0]]
            sh1_2d   = tri_2d[shared[1]]
            d0 = np.linalg.norm(free_2d - sh0_2d)
            d1 = np.linalg.norm(free_2d - sh1_2d)

            # Fold angle determines the direction of the new face
            # relative to the current face normal
            fold_angle = fe.dihedral_angle

            # The new face normal is rotated from current face normal
            # around the shared edge by (pi - dihedral_angle)
            edge_vec  = p1_3d - p0_3d
            edge_unit = edge_vec / np.linalg.norm(edge_vec)

            # Rodrigues rotation of face normal around edge
            theta = -(np.pi - fold_angle) * direction
            c, s  = np.cos(theta), np.sin(theta)
            new_normal = (c * face_normal
                          + s * np.cross(edge_unit, face_normal)
                          + (1-c) * np.dot(edge_unit, face_normal) * edge_unit)

            # Place free vertex using new face normal as reference plane
            free_3d = place_free_vertex(
                p0_3d, p1_3d, d0, d1, new_normal, 1, fold_angle
            )

            neighbour_3d = np.zeros((3, 3))
            neighbour_3d[shared[0]] = p0_3d
            neighbour_3d[shared[1]] = p1_3d
            neighbour_3d[free]      = free_3d

            placed[neighbour_id] = neighbour_3d
            queue.append(neighbour_id)

    return placed


def verify_closure(nf: NetFold, placed: dict, tol: float = 1e-6) -> list:
    """
    Check that stitch edges meet in 3D after reconstruction.
    """
    results = []
    for se in nf.stitch_edges:
        if se.tri_a not in placed or se.tri_b not in placed:
            results.append({
                'stitch': (se.tri_a, se.tri_b),
                'error': None,
                'status': 'NOT PLACED'
            })
            continue

        verts_a = placed[se.tri_a]
        verts_b = placed[se.tri_b]

        p0_a = verts_a[se.local_a[0]]
        p1_a = verts_a[se.local_a[1]]
        p0_b = verts_b[se.local_b[0]]
        p1_b = verts_b[se.local_b[1]]

        err = min(
            np.linalg.norm(p0_a - p0_b) + np.linalg.norm(p1_a - p1_b),
            np.linalg.norm(p0_a - p1_b) + np.linalg.norm(p1_a - p0_b)
        )
        results.append({
            'stitch': (se.tri_a, se.tri_b),
            'error': round(err, 8),
            'status': 'OK' if err < tol else 'MISMATCH'
        })

    return results