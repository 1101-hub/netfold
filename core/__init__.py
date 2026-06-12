"""
NetFold — A geometry format for folded/fabricated structures.

Public API:
    from core import NetFold, encode_mesh, reconstruct, save_netfold, load_netfold
"""

from .structure import (
    NetFold,
    Triangle2D,
    FoldEdge,
    StitchEdge,
    CutEdge,
    RootAnchor,
)
from .auto_encoder import encode_mesh, encode_mesh_multi_island
from .reconstruct import reconstruct, verify_closure, refine_closure, denormalize
from .serialise import (
    save_netfold, load_netfold,
    save_netfold_json, load_netfold_json,
    save_netfold_binary, load_netfold_binary,
)
from .overlap import find_overlaps, sat_intersect
from .export_svg import export_svg

__all__ = [
    "NetFold",
    "Triangle2D",
    "FoldEdge",
    "StitchEdge",
    "CutEdge",
    "RootAnchor",
    "encode_mesh",
    "encode_mesh_multi_island",
    "reconstruct",
    "verify_closure",
    "refine_closure",
    "denormalize",
    "save_netfold",
    "load_netfold",
    "save_netfold_json",
    "load_netfold_json",
    "save_netfold_binary",
    "load_netfold_binary",
    "find_overlaps",
    "sat_intersect",
    "export_svg",
]
