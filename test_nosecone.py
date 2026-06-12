"""Test nosecone with different roots and refine_closure iterations."""
import sys
sys.path.insert(0, '.')
from core.auto_encoder import encode_mesh
from core.reconstruct import reconstruct, verify_closure, refine_closure

def test_root(root_idx, label=""):
    nf = encode_mesh('test/nosecone.obj', name='nosecone', root_tri_idx=root_idx)
    placed = reconstruct(nf)
    results = verify_closure(nf, placed)
    errs = [r['error'] for r in results if r['error'] is not None]
    max_err = max(errs) if errs else 0
    n_ok = sum(1 for r in results if r['status'] == 'OK')

    # Refine
    placed2 = refine_closure(nf, placed, iterations=200, damping=0.5)
    results2 = verify_closure(nf, placed2)
    errs2 = [r['error'] for r in results2 if r['error'] is not None]
    max_err2 = max(errs2) if errs2 else 0
    n_ok2 = sum(1 for r in results2 if r['status'] == 'OK')

    print(f"Root {root_idx:4d} {label:20s}  "
          f"raw={max_err:.3f} ok={n_ok}/{len(results)}  "
          f"refined={max_err2:.3f} ok={n_ok2}/{len(results2)}")

# Test several roots
for idx in [0, 100, 200, 300, 340, 360, 400, 500, 600, 700, 719]:
    test_root(idx, "")
