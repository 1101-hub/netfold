"""
NetFold Auto-Encoder
====================
OBJ triangulated mesh → NetFold encoding.

Conventions (matched to reconstruct.py):
  dihedral_angle : interior dihedral in radians
                   π  = flat (coplanar triangles within a face)
                   π/2 = 90° fold (adjacent cube faces)
  local_a        : (la, (la+1)%3)  — edge vertex indices in tri_a's winding order
  local_b        : ((lb+1)%3, lb)  — REVERSED so local_b[0] maps to same 3D vertex as local_a[0]
  2D positions   : computed by BFS unfolding via trilateration (edge lengths only)

Author: Amulya
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

from .structure import Triangle2D, FoldEdge, StitchEdge, CutEdge, RootAnchor, NetFold
from .optimal_root import optimal_root, optimal_root_fast, optimal_root_endpoint


# ── OBJ parser ────────────────────────────────────────────────────────────────

def parse_obj(path: str | Path):
    """Returns (vertices: float64 [N,3], faces: list of [v0,v1,v2] 0-indexed)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'v':
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == 'f':
                idx = [int(t.split('/')[0]) - 1 for t in tok[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float64), faces


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Near-zero vector: {v}")
    return v / n


def _face_normal(vertices: np.ndarray, face: list) -> np.ndarray:
    """Unnormalized normal via (b-a) × (c-a)."""
    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    return np.cross(b - a, c - a)


# ── Edge map ──────────────────────────────────────────────────────────────────

def _build_edge_map(faces: list) -> dict:
    """{(v_min, v_max): [(tri_idx, local_edge_idx), ...]}"""
    em = defaultdict(list)
    for ti, tri in enumerate(faces):
        for le in range(3):
            v0, v1 = tri[le], tri[(le + 1) % 3]
            em[(min(v0, v1), max(v0, v1))].append((ti, le))
    return em


# ── Orientation propagation ───────────────────────────────────────────────────

def _propagate_orientations(faces: list, edge_map_raw: dict) -> list:
    """
    BFS from face 0 to make all windings consistent (outward normals).
    For each shared edge, enforces opposite traversal directions in the two triangles.
    Returns a new face list — does not mutate input.
    """
    faces = [list(f) for f in faces]
    n = len(faces)
    visited = [False] * n
    visited[0] = True
    queue = deque([0])

    while queue:
        cur = queue.popleft()
        for le in range(3):
            v0, v1 = faces[cur][le], faces[cur][(le + 1) % 3]
            key = (min(v0, v1), max(v0, v1))
            for nbr, nbr_le in edge_map_raw.get(key, []):
                if nbr == cur or visited[nbr]:
                    continue
                visited[nbr] = True
                # Consistent = opposite traversal. Same direction → flip nbr.
                if faces[nbr][nbr_le] == v0 and faces[nbr][(nbr_le + 1) % 3] == v1:
                    faces[nbr] = [faces[nbr][0], faces[nbr][2], faces[nbr][1]]
                queue.append(nbr)

    return faces


# ── Fold edge parameters ──────────────────────────────────────────────────────

def _compute_fold_params(
    vertices: np.ndarray,
    faces: list,
    ta_idx: int,
    tb_idx: int,
    la: int,
    lb: int,
) -> tuple[float, int]:
    """
    Returns (dihedral_interior, fold_direction) for the spanning edge ta → tb.

    dihedral_interior = π - angle_between_outward_normals
      Matches the convention in encoder.py (FOLD_180, FOLD_90).
      The reconstructor computes theta = direction * (π - dihedral_angle),
      which gives the correct fold rotation.

    fold_direction = +1 if cross(u_3d, v_perp_nbr) · na > 0 else -1
      u_3d = directed along ta[la] → ta[(la+1)%3] (ta's winding).
      v_perp_nbr = component of (tb's free vertex - p0) perp to u_3d.
      Consistent across all edges of a face for consistently-wound meshes.
    """
    ta, tb = faces[ta_idx], faces[tb_idx]
    na = _unit(_face_normal(vertices, ta))
    nb = _unit(_face_normal(vertices, tb))

    p0 = vertices[ta[la]]
    p1 = vertices[ta[(la + 1) % 3]]
    u_3d = _unit(p1 - p0)

    opp_b = vertices[tb[(lb + 2) % 3]]
    v_raw = opp_b - p0
    v_perp = v_raw - np.dot(v_raw, u_3d) * u_3d

    fold_direction = 1 if float(np.dot(np.cross(u_3d, v_perp), na)) > 0 else -1

    cos_d = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    sin_d = float(np.dot(np.cross(na, nb), u_3d))
    dihedral_interior = np.pi - float(np.arctan2(sin_d, cos_d))

    return dihedral_interior, fold_direction


# ── 2D net layout ─────────────────────────────────────────────────────────────

def _root_face_2d(vertices: np.ndarray, face: list) -> np.ndarray:
    """
    Place root face in 2D using edge lengths only (no fold angles needed):
      local vertex 0 → (0, 0)
      local vertex 1 → (d01, 0)   along +x
      local vertex 2 → trilaterated into upper half-plane
    Returns shape (3, 2).
    """
    p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    d01 = float(np.linalg.norm(p1 - p0))
    d02 = float(np.linalg.norm(p2 - p0))
    d12 = float(np.linalg.norm(p2 - p1))

    along = (d02 ** 2 - d12 ** 2 + d01 ** 2) / (2.0 * d01)
    perp  = float(np.sqrt(max(0.0, d02 ** 2 - along ** 2)))

    return np.array([
        [0.0,   0.0  ],
        [d01,   0.0  ],
        [along, perp ],
    ])


def _child_face_2d(
    parent_2d: np.ndarray,
    la: int,
    lb: int,
    vertices: np.ndarray,
    parent_face: list,
    child_face: list,
) -> np.ndarray:
    """
    Unfold child triangle into 2D net by trilateration.

    After propagation the shared edge is traversed as:
      ta: ta[la] → ta[(la+1)%3]  = v0 → v1
      tb: tb[lb] → tb[(lb+1)%3] = v1 → v0  (opposite)

    So:
      child_2d[(lb+1)%3] = parent_2d[la]       = v0  (local_b[0] → same 3D vertex as local_a[0])
      child_2d[lb]       = parent_2d[(la+1)%3] = v1  (local_b[1] → same 3D vertex as local_a[1])
      child_2d[(lb+2)%3] = trilaterated free vertex, on opposite side from parent's free vertex
    """
    v0_2d = parent_2d[la]
    v1_2d = parent_2d[(la + 1) % 3]

    edge     = v1_2d - v0_2d
    edge_len = float(np.linalg.norm(edge))
    u_2d     = edge / edge_len
    vp_2d    = np.array([-u_2d[1], u_2d[0]])   # CCW perpendicular

    # 3D vertex indices for distance computation.
    # After propagation: parent_face[la] = v0 = child_face[(lb+1)%3]
    v0_3d_idx   = parent_face[la]
    v1_3d_idx   = parent_face[(la + 1) % 3]
    free_3d_idx = child_face[(lb + 2) % 3]

    d_v0 = float(np.linalg.norm(vertices[free_3d_idx] - vertices[v0_3d_idx]))
    d_v1 = float(np.linalg.norm(vertices[free_3d_idx] - vertices[v1_3d_idx]))

    # Trilaterate free vertex from v0_2d along u_2d
    along   = (d_v0 ** 2 - d_v1 ** 2 + edge_len ** 2) / (2.0 * edge_len)
    perp    = float(np.sqrt(max(0.0, d_v0 ** 2 - along ** 2)))

    # Place on opposite side from parent's free vertex
    parent_free_2d   = parent_2d[(la + 2) % 3]
    parent_perp_sign = float(np.dot(parent_free_2d - v0_2d, vp_2d))
    free_sign        = -1.0 if parent_perp_sign >= 0.0 else 1.0

    free_2d = v0_2d + along * u_2d + free_sign * perp * vp_2d

    child_2d = np.zeros((3, 2))
    child_2d[(lb + 1) % 3] = v0_2d    # local_b[0] → v0  (same 3D vertex as local_a[0])
    child_2d[lb]           = v1_2d    # local_b[1] → v1  (same 3D vertex as local_a[1])
    child_2d[(lb + 2) % 3] = free_2d

    return child_2d


# ── Main encoder ──────────────────────────────────────────────────────────────

def encode_mesh(
    obj_path: str | Path,
    name: str = "mesh",
    root_tri_idx: int = 0,
    normalize: bool = False,
) -> NetFold:
    """
    Full pipeline: OBJ → NetFold.

    obj_path     : path to .obj file (triangulated; quads are fan-triangulated)
    name         : label stored in NetFold.name
    root_tri_idx : BFS root triangle. Default 0 is fine for small meshes.
                   For large meshes, pick the face closest to the mesh centroid
                   to minimise max fold-chain depth and float error accumulation.
    normalize    : if True, scale mesh to unit bounding box before encoding.
                   Original coordinates can be recovered via:
                     original = reconstructed * nf.scale + nf.translate

    Returns a NetFold with build_adjacency() already called.
    """
    vertices, faces_raw = parse_obj(obj_path)

    # Optional normalisation to unit bounding box
    norm_scale, norm_translate = None, None
    if normalize:
        norm_translate = vertices.min(axis=0).copy()
        vertices = vertices - norm_translate
        norm_scale = float(vertices.max())
        if norm_scale > 1e-12:
            vertices = vertices / norm_scale
        else:
            norm_scale = 1.0

    # Validate manifold closed mesh
    edge_map_raw = _build_edge_map(faces_raw)
    non_manifold = [k for k, v in edge_map_raw.items() if len(v) > 2]
    boundary     = [k for k, v in edge_map_raw.items() if len(v) == 1]
    if non_manifold:
        raise ValueError(
            f"Non-manifold mesh: {len(non_manifold)} edges shared by 3+ triangles."
        )
    if boundary:
        raise ValueError(
            f"Open mesh: {len(boundary)} boundary edges. NetFold requires a closed surface."
        )

    # Propagate consistent outward winding, then rebuild edge map
    faces    = _propagate_orientations(faces_raw, edge_map_raw)
    edge_map = _build_edge_map(faces)

    # Dual graph adjacency: ti → [(nbr, la, lb), ...]
    dual_adj: dict[int, list] = defaultdict(list)
    for edge_key, tris in edge_map.items():
        if len(tris) == 2:
            (ta, la), (tb, lb) = tris
            dual_adj[ta].append((tb, la, lb))
            dual_adj[tb].append((ta, lb, la))

    # Auto-select root if caller left the default
    if root_tri_idx == 0:
        n_tris = len(faces)
        if n_tris <= 100:
            root_tri_idx, _ = optimal_root(dual_adj, n_tris)
        elif n_tris <= 2000:
            # For larger meshes use the diameter endpoint — reduces max stitch
            # error accumulation on elongated meshes (nose cones, cylinders).
            root_tri_idx, _ = optimal_root_endpoint(dual_adj, n_tris)
        else:
            root_tri_idx, _ = optimal_root_fast(dual_adj, n_tris)

    # BFS: spanning tree + 2D positions
    n_tris       = len(faces)
    positions_2d = [None] * n_tris
    positions_2d[root_tri_idx] = _root_face_2d(vertices, faces[root_tri_idx])

    visited = [False] * n_tris
    visited[root_tri_idx] = True
    queue = deque([root_tri_idx])

    spanning_canon: set[frozenset] = set()
    fold_edges:  list[FoldEdge]   = []
    stitch_candidates: list       = []

    while queue:
        cur = queue.popleft()
        for nbr, la, lb in dual_adj[cur]:
            canon = frozenset([cur, nbr])
            if not visited[nbr]:
                visited[nbr] = True
                spanning_canon.add(canon)



                dihedral, fold_dir = _compute_fold_params(
                    vertices, faces, cur, nbr, la, lb
                )
                fold_edges.append(FoldEdge(
                    tri_a=cur,
                    tri_b=nbr,
                    local_a=(la, (la + 1) % 3),
                    local_b=((lb + 1) % 3, lb),   # reversed: local_b[0] = same 3D vertex as local_a[0]
                    dihedral_angle=dihedral,
                    fold_direction=fold_dir,
                ))

                positions_2d[nbr] = _child_face_2d(
                    positions_2d[cur], la, lb,
                    vertices, faces[cur], faces[nbr]
                )
                queue.append(nbr)
            else:
                stitch_candidates.append((cur, nbr, la, lb, canon))

    # Deduplicate stitch edges (each non-tree edge appears twice in dual_adj)
    seen_stitch: set[frozenset] = set()
    stitch_edges: list[StitchEdge] = []
    for cur, nbr, la, lb, canon in stitch_candidates:
        if canon in spanning_canon or canon in seen_stitch:
            continue
        seen_stitch.add(canon)
        stitch_edges.append(StitchEdge(
            tri_a=cur,
            tri_b=nbr,
            local_a=(la, (la + 1) % 3),
            local_b=(lb, (lb + 1) % 3),
            dihedral_angle=0.0,
        ))

    # Triangle2D list — index i must have id=i (reconstruct.py uses list index as ID)
    triangles = [
        Triangle2D(id=i, vertices=positions_2d[i])
        for i in range(n_tris)
    ]

    # Root anchor: 3D vertex positions of root face
    rf = faces[root_tri_idx]
    root = RootAnchor(
        triangle_id=root_tri_idx,
        position_3d=np.array([vertices[rf[0]], vertices[rf[1]], vertices[rf[2]]]),
        normal_3d=_unit(_face_normal(vertices, rf)),
    )

    nf = NetFold(
        name=name,
        triangles=triangles,
        fold_edges=fold_edges,
        stitch_edges=stitch_edges,
        root=root,
        scale=norm_scale,
        translate=norm_translate,
    )
    nf.build_adjacency()
    return nf


# ── Multi-island encoder ──────────────────────────────────────────────────────

def _bfs_island(
    root_idx: int,
    vertices: np.ndarray,
    faces: list,
    dual_adj: dict,
    forbidden_edges: set,        # frozenset pairs of (ta, tb) not to cross
) -> tuple:
    """
    BFS from root_idx, skipping forbidden edges.
    Returns (positions_2d, fold_edges, stitch_candidates, visited_set).
    """
    n_tris = len(faces)
    positions_2d = [None] * n_tris
    positions_2d[root_idx] = _root_face_2d(vertices, faces[root_idx])

    visited = set([root_idx])
    queue   = deque([root_idx])

    spanning_canon: set = set()
    fold_edges: list    = []
    stitch_candidates   = []

    while queue:
        cur = queue.popleft()
        for nbr, la, lb in dual_adj[cur]:
            canon = frozenset([cur, nbr])
            if canon in forbidden_edges:
                continue
            if not visited.issuperset({nbr}) is False:
                pass
            if nbr not in visited:
                visited.add(nbr)
                spanning_canon.add(canon)
                dihedral, fold_dir = _compute_fold_params(
                    vertices, faces, cur, nbr, la, lb
                )
                fold_edges.append(FoldEdge(
                    tri_a=cur, tri_b=nbr,
                    local_a=(la, (la + 1) % 3),
                    local_b=((lb + 1) % 3, lb),
                    dihedral_angle=dihedral,
                    fold_direction=fold_dir,
                ))
                positions_2d[nbr] = _child_face_2d(
                    positions_2d[cur], la, lb,
                    vertices, faces[cur], faces[nbr]
                )
                queue.append(nbr)
            else:
                stitch_candidates.append((cur, nbr, la, lb, canon))

    return positions_2d, fold_edges, stitch_candidates, spanning_canon, visited


def _offset_island(positions_2d: list, tri_ids: list, x_offset: float) -> float:
    """
    Translate all triangles in the island rightward by x_offset.
    Returns the new x_offset (right edge of this island + padding).
    """
    verts = [positions_2d[i] for i in tri_ids]
    if not verts:
        return x_offset

    all_pts = np.vstack(verts)
    min_x   = all_pts[:, 0].min()
    max_x   = all_pts[:, 0].max()
    shift   = x_offset - min_x + 0.05  # small padding

    for i in tri_ids:
        positions_2d[i] = positions_2d[i] + np.array([shift, 0.0])

    return x_offset + (max_x - min_x) + 0.15   # gap between islands


def encode_mesh_multi_island(
    obj_path: str | Path,
    name: str = "mesh",
    normalize: bool = False,
    max_iterations: int = 300,
) -> NetFold:
    """
    Overlap-aware encoder: OBJ → multi-island NetFold.

    Iteratively:
      1. BFS unfold from optimal root (skipping cut edges)
      2. Detect 2D overlaps with SAT
      3. For each overlapping pair, cut the fold edge connecting
         the deeper triangle to its BFS parent
      4. Re-run BFS — the cut triangle becomes the root of a new island
      5. Repeat until no overlaps remain or max_iterations reached

    Each island is placed side-by-side in 2D (no overlap by construction).

    Parameters:
        obj_path       : path to .obj file
        name           : label stored in NetFold.name
        normalize      : if True, scale to unit bounding box
        max_iterations : safety cap on the cut-and-retry loop

    Returns:
        NetFold with build_adjacency() called.
        nf.cut_edges  — edges that were severed
        nf.islands    — list of triangle ID lists, one per island
    """
    from .overlap import find_overlaps, OverlapResult

    vertices, faces_raw = parse_obj(obj_path)

    norm_scale, norm_translate = None, None
    if normalize:
        norm_translate = vertices.min(axis=0).copy()
        vertices = vertices - norm_translate
        norm_scale = float(vertices.max())
        if norm_scale > 1e-12:
            vertices = vertices / norm_scale
        else:
            norm_scale = 1.0

    # Validate
    edge_map_raw = _build_edge_map(faces_raw)
    non_manifold = [k for k, v in edge_map_raw.items() if len(v) > 2]
    boundary     = [k for k, v in edge_map_raw.items() if len(v) == 1]
    if non_manifold:
        raise ValueError(f"Non-manifold mesh: {len(non_manifold)} edges shared by 3+ triangles.")
    if boundary:
        raise ValueError(f"Open mesh: {len(boundary)} boundary edges. NetFold requires a closed surface.")

    faces    = _propagate_orientations(faces_raw, edge_map_raw)
    edge_map = _build_edge_map(faces)
    n_tris   = len(faces)

    # Build dual graph
    dual_adj: dict = defaultdict(list)
    for edge_key, tris in edge_map.items():
        if len(tris) == 2:
            (ta, la), (tb, lb) = tris
            dual_adj[ta].append((tb, la, lb))
            dual_adj[tb].append((ta, lb, la))

    # BFS depth map helper (used to pick which triangle to cut)
    def _bfs_depth(root, forbidden):
        depth = {root: 0}
        q = deque([root])
        while q:
            cur = q.popleft()
            for nbr, *_ in dual_adj[cur]:
                if frozenset([cur, nbr]) in forbidden:
                    continue
                if nbr not in depth:
                    depth[nbr] = depth[cur] + 1
                    q.append(nbr)
        return depth

    # Choose global root — use endpoint for large meshes
    if n_tris <= 100:
        root_tri_idx, _ = optimal_root(dual_adj, n_tris)
    elif n_tris <= 2000:
        root_tri_idx, _ = optimal_root_endpoint(dual_adj, n_tris)
    else:
        root_tri_idx, _ = optimal_root_fast(dual_adj, n_tris)

    # ── Iterative cut loop — smart bulk strategy ──────────────────────────────
    forbidden_edges: set = set()   # frozensets of (ta, tb) cut so far
    cut_metadata: list  = []       # (ta, tb, la, lb, dihedral, fold_dir)

    positions_2d     = [None] * n_tris
    all_fold_edges:  list = []
    island_roots:    list = [root_tri_idx]

    def _do_bfs_pass():
        """Run full BFS over all island roots. Returns all state."""
        pos2d        = [None] * n_tris
        fold_edges_  = []
        stitch_cands = []
        spanning_    : set = set()
        isl_list     = []
        visited_g    : set = set()

        for i_root in island_roots:
            if i_root in visited_g:
                cands = [t for t in range(n_tris) if t not in visited_g]
                if not cands:
                    break
                i_root = cands[0]
            p, fe, sc, sp, vis = _bfs_island(
                i_root, vertices, faces, dual_adj, forbidden_edges
            )
            isl_list.append(sorted(vis))
            for tid in vis:
                if p[tid] is not None:
                    pos2d[tid] = p[tid]
            fold_edges_.extend(fe)
            stitch_cands.extend(sc)
            spanning_ |= sp
            visited_g  |= vis

        # Mop up any disconnected triangles
        for t in range(n_tris):
            if t not in visited_g:
                p, fe, sc, sp, vis = _bfs_island(
                    t, vertices, faces, dual_adj, forbidden_edges
                )
                isl_list.append(sorted(vis))
                for tid in vis:
                    if p[tid] is not None:
                        pos2d[tid] = p[tid]
                fold_edges_.extend(fe)
                stitch_cands.extend(sc)
                spanning_ |= sp
                visited_g  |= vis
                island_roots.append(t)

        # Offset islands side by side
        x_cur = 0.0
        for isl in isl_list:
            x_cur = _offset_island(pos2d, isl, x_cur)

        return pos2d, fold_edges_, stitch_cands, spanning_, isl_list

    def _record_cut(fe, fold_edges_local):
        """Mark fe as forbidden and record it in cut_metadata. Returns True."""
        canon = frozenset([fe.tri_a, fe.tri_b])
        if canon in forbidden_edges:
            return False
        forbidden_edges.add(canon)
        cut_metadata.append((fe.tri_a, fe.tri_b, fe.local_a, fe.local_b,
                             fe.dihedral_angle, fe.fold_direction))
        if fe.tri_b not in island_roots:
            island_roots.append(fe.tri_b)
        return True

    for iteration in range(max_iterations + 1):
        positions_2d, all_fold_edges, all_stitch_candidates, \
            all_spanning_canon, islands_list = _do_bfs_pass()

        # Last iteration — accept whatever we have
        if iteration == max_iterations:
            break

        # ── Overlap check ──────────────────────────────────────────────────
        tmp_triangles = [Triangle2D(id=i, vertices=positions_2d[i])
                         for i in range(n_tris)]
        rf = faces[root_tri_idx]
        tmp_root = RootAnchor(
            triangle_id=root_tri_idx,
            position_3d=np.array([vertices[rf[0]], vertices[rf[1]], vertices[rf[2]]]),
            normal_3d=_unit(_face_normal(vertices, rf)),
        )
        tmp_nf = NetFold(name="_tmp", triangles=tmp_triangles,
                         fold_edges=all_fold_edges, stitch_edges=[], root=tmp_root)
        tmp_nf.build_adjacency()

        from .overlap import find_overlaps
        overlap_result = find_overlaps(tmp_nf)

        if overlap_result.clean:
            break

        # ── Build depth map ────────────────────────────────────────────────
        depth_map = _bfs_depth(root_tri_idx, forbidden_edges)

        # ── Parent map: tri_b → fold_edge that placed it ───────────────────
        parent_fe: dict = {}   # tri_b → FoldEdge
        for fe in all_fold_edges:
            if fe.tri_b not in parent_fe:
                parent_fe[fe.tri_b] = fe

        # ── Strategy 1: BULK RING CUT ──────────────────────────────────────
        # Collect all overlapping triangles, group by BFS depth.
        # If a depth-ring has many overlapping members, cut the entire ring
        # (all its parent fold edges) at once — converts nosecone to segments.
        overlap_tris = set()
        for (oa, ob) in overlap_result.pairs:
            overlap_tris.add(oa)
            overlap_tris.add(ob)

        # Depth → set of overlapping triangles at that depth
        depth_groups: dict = {}
        for t in overlap_tris:
            d = depth_map.get(t, 0)
            depth_groups.setdefault(d, set()).add(t)

        # Count triangles at each depth total
        depth_total: dict = {}
        for t in range(n_tris):
            d = depth_map.get(t, 0)
            depth_total[d] = depth_total.get(d, 0) + 1

        # Find the deepest depth where ≥50% of the ring is overlapping
        bulk_cuts_made = 0
        best_depth = -1
        best_ratio = 0.0
        for d, ovlp_set in depth_groups.items():
            total = depth_total.get(d, 1)
            ratio = len(ovlp_set) / total
            if ratio > best_ratio:
                best_ratio = ratio
                best_depth = d

        if best_ratio >= 0.4 and best_depth > 0:
            # Cut ALL triangles at this depth from their parents
            tris_at_depth = [t for t in range(n_tris)
                             if depth_map.get(t, 0) == best_depth]
            for t in tris_at_depth:
                fe = parent_fe.get(t)
                if fe is not None:
                    if _record_cut(fe, all_fold_edges):
                        bulk_cuts_made += 1

        if bulk_cuts_made > 0:
            continue   # re-run BFS with ring cut applied

        # ── Strategy 2: FAN ROOT CUT ───────────────────────────────────────
        # For each overlapping triangle, find the ancestor with the most
        # overlapping descendants. Cut at that ancestor's parent edge
        # (splits a fan into two separate subtrees).
        ancestor_overlap_count: dict = {}   # ancestor_tri → count
        for t in overlap_tris:
            # Walk up the BFS tree to root, increment each ancestor's count
            cur = t
            visited_anc = set()
            while cur in parent_fe and cur not in visited_anc:
                visited_anc.add(cur)
                anc = parent_fe[cur].tri_a
                ancestor_overlap_count[anc] = ancestor_overlap_count.get(anc, 0) + 1
                cur = anc

        if ancestor_overlap_count:
            # Pick the ancestor with most overlapping descendants
            best_anc = max(ancestor_overlap_count, key=ancestor_overlap_count.get)
            fe = parent_fe.get(best_anc)
            if fe is not None and _record_cut(fe, all_fold_edges):
                continue

        # ── Strategy 3: SINGLE EDGE CUT (fallback) ────────────────────────
        cut_happened = False
        for (oa, ob) in overlap_result.pairs:
            deeper = oa if depth_map.get(oa, 0) >= depth_map.get(ob, 0) else ob
            fe = parent_fe.get(deeper)
            if fe is not None and _record_cut(fe, all_fold_edges):
                cut_happened = True
                break

        if not cut_happened:
            break   # no more cuttable edges

    # ── Build final stitch + cut edges ────────────────────────────────────────
    seen_stitch: set   = set()
    stitch_edges: list = []
    for cur, nbr, la, lb, canon in all_stitch_candidates:
        if canon in all_spanning_canon or canon in seen_stitch:
            continue
        seen_stitch.add(canon)
        stitch_edges.append(StitchEdge(
            tri_a=cur, tri_b=nbr,
            local_a=(la, (la + 1) % 3),
            local_b=(lb, (lb + 1) % 3),
            dihedral_angle=0.0,
        ))

    # Convert cut metadata → CutEdge objects
    cut_edges: list = []
    for (ta, tb, local_a, local_b, dihedral, fold_dir) in cut_metadata:
        cut_edges.append(CutEdge(
            tri_a=ta, tri_b=tb,
            local_a=local_a, local_b=local_b,
            dihedral_angle=dihedral,
            fold_direction=fold_dir,
        ))
        # Cut edges also become stitch edges (they must meet in 3D)
        canon = frozenset([ta, tb])
        if canon not in seen_stitch:
            seen_stitch.add(canon)
            stitch_edges.append(StitchEdge(
                tri_a=ta, tri_b=tb,
                local_a=local_a,
                local_b=(local_b[1], local_b[0]),  # reversed convention
                dihedral_angle=dihedral,
            ))

    triangles = [Triangle2D(id=i, vertices=positions_2d[i]) for i in range(n_tris)]

    rf   = faces[root_tri_idx]
    root = RootAnchor(
        triangle_id=root_tri_idx,
        position_3d=np.array([vertices[rf[0]], vertices[rf[1]], vertices[rf[2]]]),
        normal_3d=_unit(_face_normal(vertices, rf)),
    )

    nf = NetFold(
        name=name,
        triangles=triangles,
        fold_edges=all_fold_edges,
        stitch_edges=stitch_edges,
        root=root,
        cut_edges=cut_edges,
        islands=islands_list,
        scale=norm_scale,
        translate=norm_translate,
    )
    nf.build_adjacency()
    return nf