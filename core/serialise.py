# core/serialise.py
import json
import numpy as np
from pathlib import Path
from .structure import NetFold, Triangle2D, FoldEdge, StitchEdge, RootAnchor

_VERSION = 1


def save_netfold(nf: NetFold, path: str | Path) -> None:
    path = Path(path).with_suffix('.netfold')
    data = {
        "version": _VERSION,
        "name":    nf.name,
        "root": {
            "triangle_id": int(nf.root.triangle_id),
            "position_3d": nf.root.position_3d.tolist(),   # (3,3)
            "normal_3d":   nf.root.normal_3d.tolist(),     # (3,)
        },
        "triangles": [
            {"id": t.id, "vertices": t.vertices.tolist()}  # (3,2)
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
    }
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


def load_netfold(path: str | Path) -> NetFold:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if data["version"] != _VERSION:
        raise ValueError(f"Version mismatch: got {data['version']}, expected {_VERSION}")

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
    nf = NetFold(
        name         = data["name"],
        triangles    = triangles,
        fold_edges   = fold_edges,
        stitch_edges = stitch_edges,
        root         = root,
    )
    nf.build_adjacency()
    return nf