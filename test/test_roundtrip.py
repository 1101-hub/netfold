"""
Round-trip test: OBJ → encode_mesh → reconstruct → verify_closure
"""

import sys, os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auto_encoder import encode_mesh
from core.reconstruct  import reconstruct, verify_closure


def count_unique_vertices(placed: dict, tol: float = 1e-6) -> int:
    all_verts = np.vstack(list(placed.values()))
    unique = [all_verts[0]]
    for v in all_verts[1:]:
        if all(np.linalg.norm(v - u) > tol for u in unique):
            unique.append(v)
    return len(unique)


def run(obj_path: str, expected_tris: int, expected_unique_verts: int,
        stitch_tol: float = 1e-6):
    name = os.path.basename(obj_path)
    print(f"\n{'='*56}")
    print(f"  {name}")
    print(f"{'='*56}")

    nf = encode_mesh(obj_path, name=name)
    nf.summary()

    assert len(nf.triangles)  == expected_tris, \
        f"Triangle count: got {len(nf.triangles)}, expected {expected_tris}"
    assert len(nf.fold_edges) == expected_tris - 1, \
        f"Fold edge count: got {len(nf.fold_edges)}, expected {expected_tris-1}"
    print(f"  Counts OK ✓")

    placed = reconstruct(nf)
    assert len(placed) == expected_tris, \
        f"Only {len(placed)}/{expected_tris} triangles placed"
    print(f"  Reconstruction placed {len(placed)}/{expected_tris} triangles ✓")

    results = verify_closure(nf, placed)
    max_err = max(r['error'] for r in results)
    all_ok  = all(r['status'] == 'OK' for r in results)
    print(f"\n  Stitch errors ({len(results)} edges):")
    for r in results:
        mark = "✓" if r['status'] == 'OK' else "✗ FAIL"
        print(f"    tris {r['stitch']}  err={r['error']:.2e}  {mark}")
    assert all_ok, f"Stitch closure failed. Max error: {max_err:.2e}"
    print(f"  All stitches closed  max_err={max_err:.2e} ✓")

    if expected_unique_verts is not None:
        n_unique = count_unique_vertices(placed)
        print(f"  Unique 3D vertices: {n_unique}  (expected {expected_unique_verts})")
        assert n_unique == expected_unique_verts
        print(f"  Unique vertices OK ✓")

    print(f"\n  PASS\n")
    return max_err


def run_geodesic(n: int, stitch_tol: float = 1e-10):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(test_dir, f"geodesic_n{n}.obj")
    if not os.path.exists(path):
        print(f"  SKIP geodesic_n{n} — file not found, run gen_geodesic.py first")
        return
    nf = encode_mesh(path, name=f"geodesic_n{n}")
    placed = reconstruct(nf)
    results = verify_closure(nf, placed)
    max_err = max(r['error'] for r in results)
    status  = "PASS ✓" if max_err < stitch_tol else "FAIL ✗"
    print(f"  geodesic n={n}  faces={len(nf.triangles):>5}  "
          f"max_stitch_err={max_err:.2e}  {status}")
    assert max_err < stitch_tol, f"geodesic n={n} stitch error {max_err:.2e} > {stitch_tol:.0e}"


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))

    run(os.path.join(test_dir, "cube.obj"),
        expected_tris=12, expected_unique_verts=8)

    run(os.path.join(test_dir, "icosahedron.obj"),
        expected_tris=20, expected_unique_verts=12)

    print(f"\n{'='*56}")
    print(f"  Geodesic spheres")
    print(f"{'='*56}")
    for n in [1, 2, 3]:
        run_geodesic(n)