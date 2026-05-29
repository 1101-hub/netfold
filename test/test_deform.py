# test/test_deform.py
"""
Parametric deformation test.
Proves: fold angles ARE the geometry.

Three experiments:
  1. Single angle sweep — one dihedral varies 0→π, shows only correct value gives zero stitch error
  2. Global lerp to flat — all angles lerp toward π, shape progressively breaks
  3. Visualise keyframes — 3D plots at 0%, 33%, 66%, 100% lerp
"""

import sys, os, copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.auto_encoder import encode_mesh
from core.reconstruct  import reconstruct, verify_closure


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def max_stitch_error(nf_mod):
    placed  = reconstruct(nf_mod)
    results = verify_closure(nf_mod, placed)
    return max(r['error'] for r in results), placed


# ── Experiment 1: single angle sweep ─────────────────────────────────────────

def exp1_single_sweep(nf_base):
    """
    Pick the first non-flat fold edge.
    Sweep its dihedral 0 → π in 200 steps.
    Plot max stitch error vs angle.
    """
    # Find first non-flat edge
    target_idx = next(
        i for i, fe in enumerate(nf_base.fold_edges)
        if abs(fe.dihedral_angle - np.pi) > 0.05
    )
    original_angle = nf_base.fold_edges[target_idx].dihedral_angle
    print(f"\nExp 1 — sweeping fold edge {target_idx}  "
          f"(original = {np.degrees(original_angle):.1f}°)")

    angles = np.linspace(0.01, np.pi - 0.01, 200)
    errors = []
    for a in angles:
        nf2 = copy.deepcopy(nf_base)
        nf2.fold_edges[target_idx].dihedral_angle = a
        err, _ = max_stitch_error(nf2)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(np.degrees(angles), errors, color='#2196F3', lw=1.5)
    ax.axvline(np.degrees(original_angle), color='#F44336', lw=1.5,
               linestyle='--', label=f'Correct angle ({np.degrees(original_angle):.1f}°)')
    ax.set_xlabel('Dihedral angle (degrees)')
    ax.set_ylabel('Max stitch error (log scale)')
    ax.set_title('Exp 1 — Single fold angle sweep\n'
                 'Only the encoded angle closes the shape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_DIR, '../output/exp1_sweep.png'), dpi=150)
    print("  Saved: output/exp1_sweep.png")
    return fig


# ── Experiment 2: global lerp to flat ────────────────────────────────────────

def exp2_global_lerp(nf_base):
    """
    Lerp ALL fold angles toward π (flat) in 50 steps.
    At t=0: original cube. At t=1: all faces coplanar — stitch error maximal.
    """
    print("\nExp 2 — global lerp of all angles toward π (flat)")
    original_angles = [fe.dihedral_angle for fe in nf_base.fold_edges]

    ts     = np.linspace(0, 1, 50)
    errors = []
    for t in ts:
        nf2 = copy.deepcopy(nf_base)
        for fe, orig in zip(nf2.fold_edges, original_angles):
            fe.dihedral_angle = orig + t * (np.pi - orig)
        err, _ = max_stitch_error(nf2)
        errors.append(err)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(ts, errors, color='#4CAF50', lw=1.5)
    ax.set_xlabel('Lerp parameter t  (0 = cube, 1 = flat)')
    ax.set_ylabel('Max stitch error (log scale)')
    ax.set_title('Exp 2 — Global lerp to flat\n'
                 'Stitch error grows as geometry deviates from encoded shape')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_DIR, '../output/exp2_lerp.png'), dpi=150)
    print("  Saved: output/exp2_lerp.png")
    return fig, ts, original_angles


# ── Experiment 3: visualise keyframes ────────────────────────────────────────

def _draw_3d(ax, placed, title, nf):
    faces_3d = list(placed.values())
    colors   = ['#FFF3CD' if tid == nf.root.triangle_id else '#4FC3F7'
                for tid in placed]
    poly = Poly3DCollection(faces_3d, alpha=0.72)
    poly.set_facecolor(colors)
    poly.set_edgecolor('#0D47A1')
    poly.set_linewidth(0.5)
    ax.add_collection3d(poly)
    all_v = np.vstack(faces_3d)
    mn, mx = all_v.min() - 0.15, all_v.max() + 0.15
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx); ax.set_zlim(mn, mx)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=22, azim=40)
    ax.set_facecolor('#F0F4F8')


def exp3_keyframes(nf_base, original_angles):
    print("\nExp 3 — 3D keyframes at t = 0, 0.33, 0.66, 1.0")
    ts_kf = [0.0, 0.33, 0.66, 1.0]
    fig   = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor('#F0F4F8')

    for i, t in enumerate(ts_kf):
        nf2 = copy.deepcopy(nf_base)
        for fe, orig in zip(nf2.fold_edges, original_angles):
            fe.dihedral_angle = orig + t * (np.pi - orig)
        _, placed = max_stitch_error(nf2)
        err, _    = max_stitch_error(nf2)

        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        _draw_3d(ax, placed,
                 f't = {t:.2f}\nerr = {err:.1e}', nf2)

    fig.suptitle('Exp 3 — Shape deformation as fold angles lerp to flat',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(TEST_DIR, '../output/exp3_keyframes.png'),
                dpi=150, bbox_inches='tight')
    print("  Saved: output/exp3_keyframes.png")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.join(TEST_DIR, '../output'), exist_ok=True)

    nf_base = encode_mesh(os.path.join(TEST_DIR, 'cube.obj'))
    err0, _ = max_stitch_error(nf_base)
    print(f"Baseline stitch error: {err0:.2e}  (should be 0)")

    fig1              = exp1_single_sweep(nf_base)
    fig2, ts, orig_a  = exp2_global_lerp(nf_base)
    fig3              = exp3_keyframes(nf_base, orig_a)

    plt.show()