"""
NetFold Stress Test

Tests the format on complex, non-convex, elongated, and genus-1 shapes.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.auto_encoder import encode_mesh_multi_island
from core.reconstruct import reconstruct, verify_closure
from core.overlap import find_overlaps

TEST_DIR = os.path.join(os.path.dirname(__file__), 'test')

MESHES = [
    ("torus", os.path.join(TEST_DIR, "torus.obj")),
    ("ellipsoid", os.path.join(TEST_DIR, "ellipsoid.obj")),
    ("sidepod", os.path.join(TEST_DIR, "sidepod.obj")),
    ("blade", os.path.join(TEST_DIR, "blade.obj"))
]

def run_stress_test():
    print("NetFold Complex Shape Stress Test")
    print("=" * 60)
    
    for name, path in MESHES:
        if not os.path.exists(path):
            print(f"Skipping {name} (not found)")
            continue
            
        print(f"\n--- {name.upper()} ---")
        try:
            # Encode
            nf = encode_mesh_multi_island(path, name=name, normalize=True)
            print(f"Encoded: {len(nf.triangles)} tris, {nf.n_islands} islands, {len(nf.cut_edges)} cut edges")
            
            # Reconstruct with auto_refine
            placed = reconstruct(nf, auto_refine=True, refine_iterations=300)
            
            # Verify
            results = verify_closure(nf, placed)
            errors = [r['error'] for r in results if r['error'] is not None]
            max_err = max(errors) if errors else 0.0
            n_ok = sum(1 for r in results if r['status'] == 'OK')
            
            # Check for overlaps
            ovlp = find_overlaps(nf)
            
            print(f"Stitches : {n_ok}/{len(results)} OK")
            print(f"Max Error: {max_err:.2e}")
            print(f"Overlaps : {ovlp.count} (clean={ovlp.clean})")
            
            if max_err < 1e-3 and ovlp.clean:
                print("Result   : PASS")
            else:
                print("Result   : FAIL")
                
        except Exception as e:
            print(f"Result   : ERROR ({e})")

if __name__ == "__main__":
    run_stress_test()
