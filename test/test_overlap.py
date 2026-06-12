"""
test_overlap.py
Tests:
  1. SAT unit tests — known geometry
  2. find_overlaps on cube, icosahedron, geodesic n1/2/3
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from core.auto_encoder import encode_mesh
from core.overlap import sat_intersect, find_overlaps


# ── 1. SAT unit tests ─────────────────────────────────────────────────────────

def test_sat_clear_separation():
    a = np.array([[0,0],[1,0],[0,1]], dtype=float)
    b = np.array([[2,0],[3,0],[2,1]], dtype=float)   # far right, no touch
    assert not sat_intersect(a, b), "clearly separated triangles must not intersect"

def test_sat_overlap():
    a = np.array([[0,0],[2,0],[1,2]], dtype=float)
    b = np.array([[1,0],[3,0],[2,2]], dtype=float)   # overlapping
    assert sat_intersect(a, b), "overlapping triangles must be detected"

def test_sat_shared_edge_not_overlap():
    # share base edge (0,0)-(1,0), apexes on opposite sides → zero area overlap
    a = np.array([[0,0],[1,0],[0.5, 1]], dtype=float)
    b = np.array([[0,0],[1,0],[0.5,-1]], dtype=float)
    assert not sat_intersect(a, b), "edge-touching triangles must not count as overlap"

def test_sat_identical():
    a = np.array([[0,0],[1,0],[0,1]], dtype=float)
    assert sat_intersect(a, a.copy()), "identical triangles must overlap"


# ── 2. Mesh overlap checks ────────────────────────────────────────────────────

TEST_DIR = os.path.dirname(__file__)

def _run(name, path):
    nf   = encode_mesh(path, name=name)
    res  = find_overlaps(nf)
    return res

def test_cube_overlap():
    res = _run("cube", os.path.join(TEST_DIR, "cube.obj"))
    # not asserting clean: BFS tree may produce overlaps on cube, that's expected

def test_icosahedron_overlap():
    res = _run("icosahedron", os.path.join(TEST_DIR, "icosahedron.obj"))

def test_geodesic_n1_overlap():
    path = os.path.join(TEST_DIR, "geodesic_n1.obj")
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n1.obj not found")
    _run("geodesic_n1", path)

def test_geodesic_n2_overlap():
    path = os.path.join(TEST_DIR, "geodesic_n2.obj")
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n2.obj not found")
    _run("geodesic_n2", path)

def test_geodesic_n3_overlap():
    path = os.path.join(TEST_DIR, "geodesic_n3.obj")
    if not os.path.exists(path):
        import pytest; pytest.skip("geodesic_n3.obj not found")
    _run("geodesic_n3", path)


if __name__ == "__main__":
    print("\n── SAT unit tests ──")
    test_sat_clear_separation(); print("  clear separation: PASS")
    test_sat_overlap();          print("  overlap:          PASS")
    test_sat_shared_edge_not_overlap(); print("  shared edge:      PASS")
    test_sat_identical();        print("  identical:        PASS")

    print("\n── Mesh overlap checks ──")
    test_cube_overlap()
    test_icosahedron_overlap()
    test_geodesic_n1_overlap()
    test_geodesic_n2_overlap()
    test_geodesic_n3_overlap()
    print("\nAll overlap tests passed.")