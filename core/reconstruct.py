"""
NetFold Reconstructor
=====================
Author: Amulya
"""

import numpy as np
from collections import deque
from .structure import NetFold

def reconstruct(
    nf: NetFold,
    auto_refine: bool = False,
    refine_iterations: int = 200,
    refine_damping: float = 0.4,
    stitch_tol: float = 1e-6,
) -> dict:
    """
    Reconstruct 3D geometry from a NetFold encoding.

    Parameters:
        nf                : the NetFold to reconstruct
        auto_refine       : if True, automatically run refine_closure when
                            stitch errors are detected. Useful for complex meshes
                            (deep BFS chains, nosecones, etc.)
        refine_iterations : max iterations for spring relaxation (if auto_refine)
        refine_damping    : damping factor for spring relaxation (if auto_refine)
        stitch_tol        : error threshold below which stitches are considered OK

    Returns:
        dict mapping triangle_id → (3,3) float64 array of 3D vertex positions
    """
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

    if auto_refine:
        # Check if refinement is needed
        results = verify_closure(nf, placed, tol=stitch_tol)
        if any(r['status'] == 'MISMATCH' for r in results):
            placed = refine_closure(nf, placed,
                                    iterations=refine_iterations,
                                    damping=refine_damping)
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


def refine_closure(
    nf: NetFold,
    placed: dict,
    iterations: int = 50,
    damping: float = 0.3,
) -> dict:
    """
    Spring-mass relaxation to reduce stitch-edge closure errors.

    For each stitch edge, computes the midpoint of mismatched vertex
    pairs and nudges both endpoints toward it. The correction is
    applied with damping to avoid oscillation.

    This is useful for large meshes where cumulative float rounding
    in the BFS reconstruction causes leaf triangles to drift.

    Parameters:
        nf         : the NetFold encoding
        placed     : dict from reconstruct() — triangle_id → (3,3) positions
        iterations : number of relaxation passes
        damping    : correction strength per iteration (0 = none, 1 = full snap)

    Returns:
        A new placed dict with reduced stitch errors.
    """
    import copy
    placed = copy.deepcopy(placed)

    for _it in range(iterations):
        # Accumulate corrections per triangle vertex
        corrections = {}  # (tri_id, vert_idx) → list of correction vectors
        for se in nf.stitch_edges:
            if se.tri_a not in placed or se.tri_b not in placed:
                continue

            verts_a = placed[se.tri_a]
            verts_b = placed[se.tri_b]

            # Determine which orientation gives less error
            err_fwd = (np.linalg.norm(verts_a[se.local_a[0]] - verts_b[se.local_b[0]]) +
                       np.linalg.norm(verts_a[se.local_a[1]] - verts_b[se.local_b[1]]))
            err_rev = (np.linalg.norm(verts_a[se.local_a[0]] - verts_b[se.local_b[1]]) +
                       np.linalg.norm(verts_a[se.local_a[1]] - verts_b[se.local_b[0]]))

            if err_fwd <= err_rev:
                pairs = [
                    ((se.tri_a, se.local_a[0]), (se.tri_b, se.local_b[0])),
                    ((se.tri_a, se.local_a[1]), (se.tri_b, se.local_b[1])),
                ]
            else:
                pairs = [
                    ((se.tri_a, se.local_a[0]), (se.tri_b, se.local_b[1])),
                    ((se.tri_a, se.local_a[1]), (se.tri_b, se.local_b[0])),
                ]

            for key_a, key_b in pairs:
                pa = placed[key_a[0]][key_a[1]]
                pb = placed[key_b[0]][key_b[1]]
                mid = (pa + pb) / 2.0

                corrections.setdefault(key_a, []).append(mid - pa)
                corrections.setdefault(key_b, []).append(mid - pb)

        if not corrections:
            break

        # Apply averaged corrections with damping
        for (tri_id, vert_idx), corr_list in corrections.items():
            avg_corr = np.mean(corr_list, axis=0)
            placed[tri_id][vert_idx] += damping * avg_corr

    return placed


def denormalize(nf: NetFold, placed: dict) -> dict:
    """
    Recover original-scale coordinates from a normalised NetFold.

    If the NetFold was encoded with normalize=True, this applies:
        original = reconstructed * scale + translate

    If the NetFold was not normalised, returns placed unchanged.
    """
    if nf.scale is None:
        return placed
    return {
        tid: verts * nf.scale + nf.translate
        for tid, verts in placed.items()
    }