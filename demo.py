"""
NetFold Demo Script
===================
Generates publication-quality visualisations of NetFold in action.
Run from the d:/netfold directory.

  python demo.py

Outputs to demo_output/:
  cube_2d.png         — 2D flat net of unit cube
  cube_3d.png         — 3D reconstruction
  ico_2d.png          — icosahedron net
  geo_2d.png          — geodesic sphere n=2 (320 triangles)
  nosecone_2d.png     — Formula Student nose cone (multi-island)
  nosecone_3d.png     — nose cone 3D reconstruction
  summary.txt         — format stats for all meshes

Author: Amulya
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.auto_encoder         import encode_mesh, encode_mesh_multi_island
from core.reconstruct          import reconstruct, verify_closure, refine_closure
from core.serialise            import save_netfold, save_netfold_binary
from core.visualise            import plot_2d_net, plot_3d_reconstruction
from core.export_svg           import export_svg

TEST_DIR   = os.path.join(os.path.dirname(__file__), 'test')
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'demo_output')
os.makedirs(OUT_DIR, exist_ok=True)

MESHES = [
    ("cube",          os.path.join(TEST_DIR, "cube.obj"),          False),
    ("icosahedron",   os.path.join(TEST_DIR, "icosahedron.obj"),   False),
    ("geodesic_n2",   os.path.join(TEST_DIR, "geodesic_n2.obj"),  False),
    ("nosecone",      os.path.join(TEST_DIR, "nosecone.obj"),      True),   # multi-island
]


def size_kb(path):
    return os.path.getsize(path) / 1024


def run_demo():
    summary_lines = [
        "NetFold Format Demo — Compression & Fabrication Stats",
        "=" * 60,
        "",
    ]

    for name, obj_path, use_multi in MESHES:
        if not os.path.exists(obj_path):
            print(f"  SKIP {name} — {obj_path} not found")
            continue

        print(f"\n{'─'*50}")
        print(f"  {name.upper()}")
        print(f"{'─'*50}")

        # ── Encode ────────────────────────────────────────────────
        if use_multi:
            nf = encode_mesh_multi_island(obj_path, name=name, normalize=True)
        else:
            nf = encode_mesh(obj_path, name=name, normalize=True)

        nf.summary()

        # ── Reconstruct + verify ───────────────────────────────────────────────
        placed    = reconstruct(nf, auto_refine=True, refine_iterations=200)
        results   = verify_closure(nf, placed)
        errors    = [r['error'] for r in results if r['error'] is not None]
        max_err   = max(errors) if errors else 0.0
        n_ok      = sum(1 for r in results if r['status'] == 'OK')
        print(f"  Stitches: {n_ok}/{len(results)} OK  |  max_err = {max_err:.2e}")

        # ── Save NetFold formats ───────────────────────────────────
        json_path = os.path.join(OUT_DIR, f"{name}.netfold")
        bin_path  = os.path.join(OUT_DIR, f"{name}.nfb")
        save_netfold(nf, json_path)
        save_netfold(nf, bin_path)

        obj_kb  = size_kb(obj_path)
        json_kb = size_kb(json_path)
        bin_kb  = size_kb(bin_path)
        print(f"  OBJ:     {obj_kb:7.1f} KB")
        print(f"  JSON:    {json_kb:7.1f} KB  ({json_kb/obj_kb*100:.0f}% of OBJ)")
        print(f"  Binary:  {bin_kb:7.1f} KB  ({bin_kb/obj_kb*100:.0f}% of OBJ)")

        # ── Visualise ─────────────────────────────────────────────
        fig2d = plot_2d_net(nf)
        fig2d.savefig(os.path.join(OUT_DIR, f"{name}_2d.png"), dpi=180, bbox_inches='tight')
        plt.close(fig2d)
        print(f"  Saved: {name}_2d.png")

        fig3d = plot_3d_reconstruction(placed, nf)
        fig3d.savefig(os.path.join(OUT_DIR, f"{name}_3d.png"), dpi=180, bbox_inches='tight')
        plt.close(fig3d)
        print(f"  Saved: {name}_3d.png")

        # Export SVG
        svg_path = os.path.join(OUT_DIR, f"{name}_pattern.svg")
        saved_svgs = export_svg(nf, svg_path)
        for p in saved_svgs:
            print(f"  Saved: {os.path.basename(p)}")

        # ── Summary entry ──────────────────────────────────────────
        summary_lines += [
            f"  {name}",
            f"    Triangles   : {len(nf.triangles)}",
            f"    Fold edges  : {len(nf.fold_edges)}",
            f"    Stitch edges: {len(nf.stitch_edges)}",
            f"    Cut edges   : {len(nf.cut_edges)}",
            f"    Islands     : {nf.n_islands}",
            f"    Stitch err  : {max_err:.2e}",
            f"    OBJ size    : {obj_kb:.1f} KB",
            f"    JSON size   : {json_kb:.1f} KB ({json_kb/obj_kb*100:.0f}%)",
            f"    Binary size : {bin_kb:.1f} KB ({bin_kb/obj_kb*100:.0f}%)",
            "",
        ]

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Summary written to {summary_path}")
    print("\nDemo complete. All outputs in demo_output/")


if __name__ == "__main__":
    run_demo()
