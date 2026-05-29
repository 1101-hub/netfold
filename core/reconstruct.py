"""
NetFold Reconstructor
=====================
Author: Amulya
"""

import numpy as np
from collections import deque
from .structure import NetFold

def reconstruct(nf: NetFold) -> dict:
    placed = {}
    root_id = nf.root.triangle_id
    placed[root_id] = nf.root.position_3d.copy()

    queue = deque([root_id])
    visited = {root_id}

    while queue:
        current_id = queue.popleft()
        current_3d = placed[current_id]

        for neighbour_id, fe in nf._adjacency[current_id]:
            if neighbour_id in visited:
                continue
            visited.add(neighbour_id)

            if fe.tri_a == current_id:
                local_current   = fe.local_a
                local_neighbour = fe.local_b
                direction       = fe.fold_direction
            else:
                local_current   = fe.local_b
                local_neighbour = fe.local_a
                direction       = -fe.fold_direction

            # Shared edge in 3D
            p0_3d = current_3d[local_current[0]]
            p1_3d = current_3d[local_current[1]]
            edge_3d = p1_3d - p0_3d
            u_3d    = edge_3d / np.linalg.norm(edge_3d)

            other_cur = [i for i in range(3) if i not in local_current][0]
            free_nbr  = [i for i in range(3) if i not in local_neighbour][0]

            cur_2d = nf.triangles[current_id].vertices
            nbr_2d = nf.triangles[neighbour_id].vertices

            # Current face 2D frame
            p0_2d_cur   = cur_2d[local_current[0]]
            p1_2d_cur   = cur_2d[local_current[1]]
            edge_2d_cur = p1_2d_cur - p0_2d_cur
            u_2d_cur    = edge_2d_cur / np.linalg.norm(edge_2d_cur)
            v_2d_cur    = np.array([-u_2d_cur[1], u_2d_cur[0]])

            other_2d  = cur_2d[other_cur]
            d_2d_cur  = other_2d - p0_2d_cur
            along_cur = np.dot(d_2d_cur, u_2d_cur)
            perp_cur  = np.dot(d_2d_cur, v_2d_cur)

            # Neighbour face 2D frame — NO antiparallel swap
            p0_2d_nbr   = nbr_2d[local_neighbour[0]]
            p1_2d_nbr   = nbr_2d[local_neighbour[1]]
            edge_2d_nbr = p1_2d_nbr - p0_2d_nbr
            u_2d_nbr    = edge_2d_nbr / np.linalg.norm(edge_2d_nbr)
            v_2d_nbr    = np.array([-u_2d_nbr[1], u_2d_nbr[0]])

            free_2d   = nbr_2d[free_nbr]
            d_2d_nbr  = free_2d - p0_2d_nbr
            along_nbr = np.dot(d_2d_nbr, u_2d_nbr)
            perp_nbr  = np.dot(d_2d_nbr, v_2d_nbr)

            # 3D perp reference from current face
            other_3d = current_3d[other_cur]
            d_3d_cur = other_3d - p0_3d
            w_3d     = d_3d_cur - along_cur * u_3d
            v_3d_ref = w_3d / perp_cur

            # Neighbour perp in unfolded position
            v_perp_nbr = perp_nbr * v_3d_ref

            # Rodrigues rotation: theta = direction * (pi - dihedral)
            theta = direction * (np.pi - fe.dihedral_angle)
            v_rot = (v_perp_nbr * np.cos(theta)
                     + np.cross(u_3d, v_perp_nbr) * np.sin(theta))

            free_3d = p0_3d + along_nbr * u_3d + v_rot

            # Assign shared vertices — NO swap, local_nbr[0] always gets p0_3d
            neighbour_3d = np.zeros((3, 3))
            neighbour_3d[local_neighbour[0]] = p0_3d
            neighbour_3d[local_neighbour[1]] = p1_3d
            neighbour_3d[free_nbr]           = free_3d

            placed[neighbour_id] = neighbour_3d
            queue.append(neighbour_id)

    return placed

def verify_closure(nf: NetFold, placed: dict, tol: float = 1e-6) -> list:
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