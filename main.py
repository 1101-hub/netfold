"""
NetFold — Proof of Concept
===========================
Stage 1: Unit cube
  1. Encode as 2D net
  2. Visualise the net
  3. Verify ground truth stitches
  4. Run reconstruction algorithm
  5. Visualise 3D result

Author: Amulya
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core.encoder     import build_cube_net, get_ground_truth_positions
from core.reconstruct import reconstruct, verify_closure
from core.visualise   import plot_2d_net, plot_3d_reconstruction

OUTPUT = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT, exist_ok=True)


def check_vertices(placed, label):
    all_verts = set()
    for verts in placed.values():
        for v in verts:
            all_verts.add(tuple(np.round(v, 3)))
    print(f"    [{label}] Unique 3D vertices: "
          f"{len(all_verts)} (expected 8)")
    for v in sorted(all_verts):
        print(f"      {v}")


def main():
    print("\n  NetFold — Proof of Concept")
    print("  " + "="*40)

    # 1. Encode
    print("\n[1] Encoding unit cube as NetFold...")
    nf = build_cube_net()
    nf.summary()

    # 2. Visualise 2D net
    print("[2] Plotting 2D net...")
    plot_2d_net(nf, save_path=os.path.join(OUTPUT, 'cube_net_2d.png'))

    # 3. Verify ground truth stitches
    print("\n[3] Verifying ground truth stitch edges...")
    gt = get_ground_truth_positions(nf)
    check_vertices(gt, "ground truth")
    report_gt = verify_closure(nf, gt)
    all_gt_ok = True
    for r in report_gt:
        mark = '✓' if r['status'] == 'OK' else '✗'
        print(f"    {mark}  stitch {r['stitch']}  "
              f"error={r['error']}  {r['status']}")
        if r['status'] != 'OK':
            all_gt_ok = False
    if all_gt_ok:
        print("\n    All ground truth stitches closed. "
              "Stitch list is correct.")
    else:
        print("\n    Ground truth stitch errors remain.")

    # 4. Reconstruction algorithm
    print("\n[4] Running reconstruction algorithm...")
    placed = reconstruct(nf)
    check_vertices(placed, "reconstructed")
    report = verify_closure(nf, placed)
    all_ok = True
    for r in report:
        mark = '✓' if r['status'] == 'OK' else '✗'
        print(f"    {mark}  stitch {r['stitch']}  "
              f"error={r['error']}  {r['status']}")
        if r['status'] != 'OK':
            all_ok = False
    if all_ok:
        print("\n    Reconstruction exact.")
    else:
        print("\n    Reconstruction has errors — "
              "algorithm needs fixing.")

    # 5. Visualise with ground truth
    print("\n[5] Plotting 3D (ground truth)...")
    plot_3d_reconstruction(gt, nf,
        save_path=os.path.join(OUTPUT, 'cube_3d.png'))

    plt.show()
    print("\n  Done.")


if __name__ == '__main__':
    main()
    import matplotlib.pyplot as plt
    from core.auto_encoder import encode_mesh
    from core.reconstruct import reconstruct
    from core.visualise import plot_2d_net, plot_3d_reconstruction

    obj = "test/icosahedron.obj"  # swap to any mesh

    nf = encode_mesh(obj)
    placed = reconstruct(nf)

    plot_2d_net(nf)
    plot_3d_reconstruction(placed, nf)

    plt.show()