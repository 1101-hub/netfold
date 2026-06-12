"""
Round-trip test: OBJ → encode_mesh → reconstruct → verify_closure
Also tests: serialisation (JSON + binary), normalisation, loop-closure refinement.
"""

import sys, os
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auto_encoder import encode_mesh, encode_mesh_multi_island
from core.overlap      import find_overlaps
from core.reconstruct  import reconstruct, verify_closure, refine_closure, denormalize
from core.serialise    import (
    save_netfold, load_netfold,
    save_netfold_json, load_netfold_json,
    save_netfold_binary, load_netfold_binary,
)


def count_unique_vertices(placed: dict, tol: float = 1e-6) -> int:
    all_verts = np.vstack(list(placed.values()))
    unique = [all_verts[0]]
    for v in all_verts[1:]:
        if all(np.linalg.norm(v - u) > tol for u in unique):
            unique.append(v)
    return len(unique)


# ── Core round-trip tests ─────────────────────────────────────────────────────

def _run_roundtrip(obj_path, expected_tris, expected_unique_verts, stitch_tol=1e-6):
    """Shared logic for round-trip testing."""
    name = os.path.basename(obj_path)
    nf = encode_mesh(obj_path, name=name)

    assert len(nf.triangles) == expected_tris, \
        f"Triangle count: got {len(nf.triangles)}, expected {expected_tris}"
    assert len(nf.fold_edges) == expected_tris - 1, \
        f"Fold edge count: got {len(nf.fold_edges)}, expected {expected_tris-1}"

    placed = reconstruct(nf)
    assert len(placed) == expected_tris, \
        f"Only {len(placed)}/{expected_tris} triangles placed"

    results = verify_closure(nf, placed)
    max_err = max(r['error'] for r in results)
    assert all(r['status'] == 'OK' for r in results), \
        f"Stitch closure failed. Max error: {max_err:.2e}"

    if expected_unique_verts is not None:
        n_unique = count_unique_vertices(placed)
        assert n_unique == expected_unique_verts, \
            f"Unique vertices: got {n_unique}, expected {expected_unique_verts}"

    return nf, placed, max_err


def test_cube_roundtrip(cube_obj):
    _run_roundtrip(cube_obj, expected_tris=12, expected_unique_verts=8)


def test_icosahedron_roundtrip(icosahedron_obj):
    _run_roundtrip(icosahedron_obj, expected_tris=20, expected_unique_verts=12)


def test_geodesic_n1(geodesic_obj):
    path = geodesic_obj(1)
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n1.obj not found")
    nf, placed, max_err = _run_roundtrip(path, expected_tris=80, expected_unique_verts=None)
    assert max_err < 1e-10, f"geodesic n=1 stitch error {max_err:.2e}"


def test_geodesic_n2(geodesic_obj):
    path = geodesic_obj(2)
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n2.obj not found")
    nf, placed, max_err = _run_roundtrip(path, expected_tris=320, expected_unique_verts=None)
    assert max_err < 1e-10, f"geodesic n=2 stitch error {max_err:.2e}"


def test_geodesic_n3(geodesic_obj):
    path = geodesic_obj(3)
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n3.obj not found")
    nf, placed, max_err = _run_roundtrip(path, expected_tris=1280, expected_unique_verts=None)
    assert max_err < 1e-10, f"geodesic n=3 stitch error {max_err:.2e}"


# ── Serialisation tests ──────────────────────────────────────────────────────

def test_json_roundtrip(cube_obj):
    """JSON serialise → deserialise → reconstruct must produce identical geometry."""
    nf = encode_mesh(cube_obj, name="cube_json_test")
    placed_orig = reconstruct(nf)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.netfold")
        save_netfold_json(nf, path)
        nf_loaded = load_netfold_json(path)

    placed_loaded = reconstruct(nf_loaded)
    for k in placed_orig:
        assert np.allclose(placed_orig[k], placed_loaded[k]), \
            f"JSON round-trip mismatch for triangle {k}"


def test_binary_roundtrip(cube_obj):
    """Binary serialise → deserialise → reconstruct must produce bit-exact geometry."""
    nf = encode_mesh(cube_obj, name="cube_binary_test")
    placed_orig = reconstruct(nf)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.nfb")
        save_netfold_binary(nf, path)
        nf_loaded = load_netfold_binary(path)

    placed_loaded = reconstruct(nf_loaded)
    for k in placed_orig:
        assert np.array_equal(placed_orig[k], placed_loaded[k]), \
            f"Binary round-trip mismatch for triangle {k} (should be bit-exact)"


def test_binary_size_smaller_than_json(icosahedron_obj):
    """Binary format should be significantly smaller than JSON."""
    nf = encode_mesh(icosahedron_obj, name="size_test")

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "test.netfold")
        bin_path  = os.path.join(tmp, "test.nfb")
        save_netfold_json(nf, json_path)
        save_netfold_binary(nf, bin_path)

        json_size = os.path.getsize(json_path)
        bin_size  = os.path.getsize(Path(bin_path).with_suffix('.nfb'))

    assert bin_size < json_size, \
        f"Binary ({bin_size}B) should be smaller than JSON ({json_size}B)"


def test_auto_detect_format(cube_obj):
    """save_netfold / load_netfold should auto-detect format by extension."""
    nf = encode_mesh(cube_obj, name="auto_detect_test")
    placed_orig = reconstruct(nf)

    with tempfile.TemporaryDirectory() as tmp:
        # JSON path
        json_path = os.path.join(tmp, "test.netfold")
        save_netfold(nf, json_path)
        nf_json = load_netfold(json_path)
        placed_json = reconstruct(nf_json)
        for k in placed_orig:
            assert np.allclose(placed_orig[k], placed_json[k])

        # Binary path
        bin_path = os.path.join(tmp, "test.nfb")
        save_netfold(nf, bin_path)
        nf_bin = load_netfold(bin_path)
        placed_bin = reconstruct(nf_bin)
        for k in placed_orig:
            assert np.array_equal(placed_orig[k], placed_bin[k])


# ── Normalisation tests ──────────────────────────────────────────────────────

def test_normalised_roundtrip(cube_obj):
    """Encoding with normalize=True and denormalizing should recover original geometry."""
    nf_raw = encode_mesh(cube_obj, name="raw", normalize=False)
    nf_norm = encode_mesh(cube_obj, name="normalised", normalize=True)

    assert nf_norm.scale is not None
    assert nf_norm.translate is not None

    placed_raw  = reconstruct(nf_raw)
    placed_norm = reconstruct(nf_norm)
    placed_denorm = denormalize(nf_norm, placed_norm)

    # After denormalisation, geometry should match the raw reconstruction
    for k in placed_raw:
        assert np.allclose(placed_raw[k], placed_denorm[k], atol=1e-10), \
            f"Denormalized triangle {k} doesn't match raw"


def test_normalised_binary_roundtrip(cube_obj):
    """Normalised NetFold survives binary serialisation."""
    nf = encode_mesh(cube_obj, name="norm_bin", normalize=True)
    placed_orig = denormalize(nf, reconstruct(nf))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.nfb")
        save_netfold_binary(nf, path)
        nf_loaded = load_netfold_binary(path)

    assert nf_loaded.scale == nf.scale
    assert np.array_equal(nf_loaded.translate, nf.translate)

    placed_loaded = denormalize(nf_loaded, reconstruct(nf_loaded))
    for k in placed_orig:
        assert np.allclose(placed_orig[k], placed_loaded[k], atol=1e-12)


# ── Loop-closure refinement test ──────────────────────────────────────────────

def test_refine_closure_does_not_increase_error(cube_obj):
    """refine_closure should never make stitch error worse."""
    nf = encode_mesh(cube_obj, name="refine_test")
    placed = reconstruct(nf)

    results_before = verify_closure(nf, placed)
    max_err_before = max(r['error'] for r in results_before)

    placed_refined = refine_closure(nf, placed, iterations=20)
    results_after = verify_closure(nf, placed_refined)
    max_err_after = max(r['error'] for r in results_after)

    assert max_err_after <= max_err_before + 1e-12, \
        f"Refinement increased error: {max_err_before:.2e} → {max_err_after:.2e}"


# ── Legacy CLI interface ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))

    cube = os.path.join(test_dir, "cube.obj")
    ico  = os.path.join(test_dir, "icosahedron.obj")

    print("\n=== Cube round-trip ===")
    nf, placed, err = _run_roundtrip(cube, 12, 8)
    print(f"  PASS  max_err={err:.2e}")

    print("\n=== Icosahedron round-trip ===")
    nf, placed, err = _run_roundtrip(ico, 20, 12)
    print(f"  PASS  max_err={err:.2e}")

    print("\n=== Geodesic spheres ===")
    for n, tris in [(1, 80), (2, 320), (3, 1280)]:
        path = os.path.join(test_dir, f"geodesic_n{n}.obj")
        if not os.path.exists(path):
            print(f"  SKIP geodesic_n{n}")
            continue
        _, _, err = _run_roundtrip(path, tris, None)
        print(f"  geodesic n={n}  faces={tris:>5}  max_err={err:.2e}  PASS")

    print("\nAll tests passed.")


# ── Multi-island tests ────────────────────────────────────────────────────────

def test_multi_island_cube_no_overlaps(cube_obj):
    """Multi-island cube should produce a clean 2D net with zero overlaps."""
    nf = encode_mesh_multi_island(cube_obj, name="cube_multi")
    nf.summary()
    assert len(nf.triangles) == 12
    # Must cover all triangles
    if nf.islands:
        covered = set(tid for isl in nf.islands for tid in isl)
        assert covered == set(range(12)), "Not all triangles covered by islands"
    # Must have zero overlaps
    from core.overlap import find_overlaps
    result = find_overlaps(nf)
    assert result.clean, f"Multi-island cube still has {result.count} overlapping pairs"


def test_multi_island_icosahedron_no_overlaps(icosahedron_obj):
    """Multi-island icosahedron should produce a clean 2D net."""
    nf = encode_mesh_multi_island(icosahedron_obj, name="ico_multi")
    assert len(nf.triangles) == 20
    result = find_overlaps(nf)
    assert result.clean, f"Multi-island icosahedron has {result.count} overlapping pairs"


def test_multi_island_reconstruction_still_valid(cube_obj):
    """Reconstruction from multi-island NetFold must still close all stitches."""
    nf = encode_mesh_multi_island(cube_obj, name="cube_reconstruct")
    placed = reconstruct(nf)
    assert len(placed) == 12
    results = verify_closure(nf, placed)
    max_err = max(r['error'] for r in results if r['error'] is not None)
    assert max_err < 1e-4, f"Multi-island reconstruction stitch error too large: {max_err:.2e}"


def test_multi_island_serialise_json(cube_obj):
    """Multi-island NetFold survives JSON round-trip preserving cut_edges and islands."""
    nf = encode_mesh_multi_island(cube_obj, name="cube_serial_json")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.netfold")
        save_netfold_json(nf, path)
        nf2 = load_netfold_json(path)
    assert len(nf2.cut_edges) == len(nf.cut_edges)
    assert len(nf2.islands)   == len(nf.islands)


def test_multi_island_serialise_binary(cube_obj):
    """Multi-island NetFold survives binary round-trip preserving cut_edges and islands."""
    nf = encode_mesh_multi_island(cube_obj, name="cube_serial_bin")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.nfb")
        save_netfold_binary(nf, path)
        nf2 = load_netfold_binary(path)
    assert len(nf2.cut_edges) == len(nf.cut_edges)
    assert len(nf2.islands)   == len(nf.islands)


def test_multi_island_geodesic_n1(geodesic_obj):
    """Geodesic n=1 multi-island should be overlap-free."""
    path = geodesic_obj(1)
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n1.obj not found")
    nf = encode_mesh_multi_island(path, name="geo1_multi")
    result = find_overlaps(nf)
    assert result.clean, f"geodesic n=1 multi-island has {result.count} overlapping pairs"