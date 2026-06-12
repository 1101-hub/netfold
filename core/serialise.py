# core/serialise.py
"""
NetFold Serialiser
==================
Two formats:
  .netfold  — JSON (human-readable, version 1)
  .nfb      — Binary (numpy npz, bit-exact, version 2)

save_netfold / load_netfold auto-detect format by extension.

Author: Amulya
"""

import json
import io
import numpy as np
from pathlib import Path
from .structure import NetFold, Triangle2D, FoldEdge, StitchEdge, CutEdge, RootAnchor

_VERSION_JSON   = 1
_VERSION_BINARY = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pack_fold_edges(fold_edges: list) -> np.ndarray:
    """Pack fold edges into a structured array for binary storage.
    Columns: tri_a, tri_b, la0, la1, lb0, lb1, dihedral_angle, fold_direction
    """
    n = len(fold_edges)
    arr = np.zeros((n, 8), dtype=np.float64)
    for i, fe in enumerate(fold_edges):
        arr[i] = [fe.tri_a, fe.tri_b,
                  fe.local_a[0], fe.local_a[1],
                  fe.local_b[0], fe.local_b[1],
                  fe.dihedral_angle, fe.fold_direction]
    return arr


def _unpack_fold_edges(arr: np.ndarray) -> list:
    edges = []
    for row in arr:
        edges.append(FoldEdge(
            tri_a=int(row[0]), tri_b=int(row[1]),
            local_a=(int(row[2]), int(row[3])),
            local_b=(int(row[4]), int(row[5])),
            dihedral_angle=float(row[6]),
            fold_direction=int(row[7]),
        ))
    return edges


def _pack_cut_edges(cut_edges: list) -> np.ndarray:
    """Same layout as fold edges: tri_a, tri_b, la0, la1, lb0, lb1, dihedral, direction"""
    n = len(cut_edges)
    if n == 0:
        return np.zeros((0, 8), dtype=np.float64)
    arr = np.zeros((n, 8), dtype=np.float64)
    for i, ce in enumerate(cut_edges):
        arr[i] = [ce.tri_a, ce.tri_b,
                  ce.local_a[0], ce.local_a[1],
                  ce.local_b[0], ce.local_b[1],
                  ce.dihedral_angle, ce.fold_direction]
    return arr


def _unpack_cut_edges(arr: np.ndarray) -> list:
    edges = []
    for row in arr:
        edges.append(CutEdge(
            tri_a=int(row[0]), tri_b=int(row[1]),
            local_a=(int(row[2]), int(row[3])),
            local_b=(int(row[4]), int(row[5])),
            dihedral_angle=float(row[6]),
            fold_direction=int(row[7]),
        ))
    return edges


def _pack_stitch_edges(stitch_edges: list) -> np.ndarray:
    """Columns: tri_a, tri_b, la0, la1, lb0, lb1, dihedral_angle"""
    n = len(stitch_edges)
    arr = np.zeros((n, 7), dtype=np.float64)
    for i, se in enumerate(stitch_edges):
        arr[i] = [se.tri_a, se.tri_b,
                  se.local_a[0], se.local_a[1],
                  se.local_b[0], se.local_b[1],
                  se.dihedral_angle]
    return arr


def _unpack_stitch_edges(arr: np.ndarray) -> list:
    edges = []
    for row in arr:
        edges.append(StitchEdge(
            tri_a=int(row[0]), tri_b=int(row[1]),
            local_a=(int(row[2]), int(row[3])),
            local_b=(int(row[4]), int(row[5])),
            dihedral_angle=float(row[6]),
        ))
    return edges


def _pack_triangles(triangles: list) -> np.ndarray:
    """Pack triangle 2D vertices into (N, 6) array: [id, x0, y0, x1, y1, x2, y2]."""
    n = len(triangles)
    arr = np.zeros((n, 7), dtype=np.float64)
    for i, t in enumerate(triangles):
        arr[i, 0] = t.id
        arr[i, 1:3] = t.vertices[0]
        arr[i, 3:5] = t.vertices[1]
        arr[i, 5:7] = t.vertices[2]
    return arr


def _unpack_triangles(arr: np.ndarray) -> list:
    tris = []
    for row in arr:
        tris.append(Triangle2D(
            id=int(row[0]),
            vertices=np.array([[row[1], row[2]],
                                [row[3], row[4]],
                                [row[5], row[6]]], dtype=np.float64),
        ))
    return tris


# ── JSON format (.netfold) ────────────────────────────────────────────────────

def save_netfold_json(nf: NetFold, path: str | Path) -> None:
    """Save NetFold to human-readable JSON (.netfold)."""
    path = Path(path).with_suffix('.netfold')
    data = {
        "version": _VERSION_JSON,
        "name":    nf.name,
        "root": {
            "triangle_id": int(nf.root.triangle_id),
            "position_3d": nf.root.position_3d.tolist(),
            "normal_3d":   nf.root.normal_3d.tolist(),
        },
        "triangles": [
            {"id": t.id, "vertices": t.vertices.tolist()}
            for t in nf.triangles
        ],
        "fold_edges": [
            {
                "tri_a":          int(fe.tri_a),
                "tri_b":          int(fe.tri_b),
                "local_a":        list(fe.local_a),
                "local_b":        list(fe.local_b),
                "dihedral_angle": float(fe.dihedral_angle),
                "fold_direction": int(fe.fold_direction),
            }
            for fe in nf.fold_edges
        ],
        "stitch_edges": [
            {
                "tri_a":          int(se.tri_a),
                "tri_b":          int(se.tri_b),
                "local_a":        list(se.local_a),
                "local_b":        list(se.local_b),
                "dihedral_angle": float(se.dihedral_angle),
            }
            for se in nf.stitch_edges
        ],
        "cut_edges": [
            {
                "tri_a":          int(ce.tri_a),
                "tri_b":          int(ce.tri_b),
                "local_a":        list(ce.local_a),
                "local_b":        list(ce.local_b),
                "dihedral_angle": float(ce.dihedral_angle),
                "fold_direction": int(ce.fold_direction),
            }
            for ce in nf.cut_edges
        ],
        "islands": nf.islands,
    }
    # Include normalisation metadata if present
    if nf.scale is not None:
        data["scale"] = float(nf.scale)
        data["translate"] = nf.translate.tolist()

    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


def load_netfold_json(path: str | Path) -> NetFold:
    """Load NetFold from JSON (.netfold)."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if data["version"] != _VERSION_JSON:
        raise ValueError(f"JSON version mismatch: got {data['version']}, expected {_VERSION_JSON}")

    root = RootAnchor(
        triangle_id = data["root"]["triangle_id"],
        position_3d = np.array(data["root"]["position_3d"], dtype=np.float64),
        normal_3d   = np.array(data["root"]["normal_3d"],   dtype=np.float64),
    )
    triangles = [
        Triangle2D(id=t["id"], vertices=np.array(t["vertices"], dtype=np.float64))
        for t in data["triangles"]
    ]
    fold_edges = [
        FoldEdge(
            tri_a          = fe["tri_a"],
            tri_b          = fe["tri_b"],
            local_a        = tuple(fe["local_a"]),
            local_b        = tuple(fe["local_b"]),
            dihedral_angle = fe["dihedral_angle"],
            fold_direction = fe["fold_direction"],
        )
        for fe in data["fold_edges"]
    ]
    stitch_edges = [
        StitchEdge(
            tri_a          = se["tri_a"],
            tri_b          = se["tri_b"],
            local_a        = tuple(se["local_a"]),
            local_b        = tuple(se["local_b"]),
            dihedral_angle = se["dihedral_angle"],
        )
        for se in data["stitch_edges"]
    ]

    # Recover normalisation metadata
    scale = data.get("scale", None)
    translate = np.array(data["translate"], dtype=np.float64) if "translate" in data else None

    # Recover cut edges (backward compat: default empty)
    cut_edges = [
        CutEdge(
            tri_a=ce["tri_a"], tri_b=ce["tri_b"],
            local_a=tuple(ce["local_a"]), local_b=tuple(ce["local_b"]),
            dihedral_angle=ce["dihedral_angle"], fold_direction=ce["fold_direction"],
        )
        for ce in data.get("cut_edges", [])
    ]
    islands = data.get("islands", [])

    nf = NetFold(
        name         = data["name"],
        triangles    = triangles,
        fold_edges   = fold_edges,
        stitch_edges = stitch_edges,
        root         = root,
        cut_edges    = cut_edges,
        islands      = islands,
        scale        = scale,
        translate    = translate,
    )
    nf.build_adjacency()
    return nf


# ── Binary format (.nfb) ─────────────────────────────────────────────────────

def save_netfold_binary(nf: NetFold, path: str | Path) -> None:
    """
    Save NetFold to compact binary format (.nfb).

    Uses numpy .npz internally. Float arrays are stored bit-exact —
    no float→str→float precision loss. ~10× smaller than JSON.
    """
    path = Path(path).with_suffix('.nfb')

    # Pack metadata as JSON bytes
    meta = {
        "version": _VERSION_BINARY,
        "name": nf.name,
        "root_triangle_id": int(nf.root.triangle_id),
    }
    if nf.scale is not None:
        meta["scale"] = float(nf.scale)
        meta["translate"] = nf.translate.tolist()

    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

    # Write to BytesIO first to avoid np.savez_compressed appending .npz
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
        root_position_3d=nf.root.position_3d,
        root_normal_3d=nf.root.normal_3d,
        triangles=_pack_triangles(nf.triangles),
        fold_edges=_pack_fold_edges(nf.fold_edges),
        stitch_edges=_pack_stitch_edges(nf.stitch_edges),
        cut_edges=_pack_cut_edges(nf.cut_edges),
    )
    # Store islands as JSON bytes in meta
    meta["islands"] = nf.islands
    # Re-encode meta with islands included
    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    buf2 = io.BytesIO()
    np.savez_compressed(
        buf2,
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
        root_position_3d=nf.root.position_3d,
        root_normal_3d=nf.root.normal_3d,
        triangles=_pack_triangles(nf.triangles),
        fold_edges=_pack_fold_edges(nf.fold_edges),
        stitch_edges=_pack_stitch_edges(nf.stitch_edges),
        cut_edges=_pack_cut_edges(nf.cut_edges),
    )
    with open(path, 'wb') as f:
        f.write(buf2.getvalue())


def load_netfold_binary(path: str | Path) -> NetFold:
    """Load NetFold from binary format (.nfb)."""
    path = Path(path)
    # Read raw bytes and load via BytesIO to handle .nfb extension
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
    data = np.load(buf, allow_pickle=False)

    meta = json.loads(data['meta'].tobytes().decode('utf-8'))
    if meta["version"] != _VERSION_BINARY:
        raise ValueError(f"Binary version mismatch: got {meta['version']}, expected {_VERSION_BINARY}")

    root = RootAnchor(
        triangle_id=meta["root_triangle_id"],
        position_3d=data['root_position_3d'].astype(np.float64),
        normal_3d=data['root_normal_3d'].astype(np.float64),
    )

    scale = meta.get("scale", None)
    translate = np.array(meta["translate"], dtype=np.float64) if "translate" in meta else None
    islands   = meta.get("islands", [])

    nf = NetFold(
        name=meta["name"],
        triangles=_unpack_triangles(data['triangles']),
        fold_edges=_unpack_fold_edges(data['fold_edges']),
        stitch_edges=_unpack_stitch_edges(data['stitch_edges']),
        cut_edges=_unpack_cut_edges(data['cut_edges']) if 'cut_edges' in data else [],
        islands=islands,
        root=root,
        scale=scale,
        translate=translate,
    )
    nf.build_adjacency()
    return nf


# ── Auto-detect wrappers ─────────────────────────────────────────────────────

def save_netfold(nf: NetFold, path: str | Path) -> None:
    """
    Save NetFold — auto-detects format by extension.
      .netfold → JSON (human-readable)
      .nfb     → binary (compact, bit-exact)
    Default: .netfold (JSON)
    """
    path = Path(path)
    if path.suffix == '.nfb':
        save_netfold_binary(nf, path)
    else:
        save_netfold_json(nf, path)


def load_netfold(path: str | Path) -> NetFold:
    """
    Load NetFold — auto-detects format by extension.
      .netfold → JSON
      .nfb     → binary
    """
    path = Path(path)
    if path.suffix == '.nfb':
        return load_netfold_binary(path)
    else:
        return load_netfold_json(path)