"""
NetFold Visualiser
==================
Two outputs:
  1. 2D net — triangles flat, fold edges annotated,
     stitch edges shown as dashed red arrows
  2. 3D reconstruction — matplotlib 3D plot

Author: Amulya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .structure import NetFold

# ── Colours ──────────────────────────────────────────────────────
C_FACE      = '#E8F4FD'
C_FACE_ROOT = '#FFF3CD'
C_FOLD_FLAT = '#ADB5BD'
C_FOLD_90   = '#2196F3'
C_STITCH    = '#F44336'
C_3D_FACE   = '#4FC3F7'
C_3D_EDGE   = '#0D47A1'
C_LABEL     = '#1A1A2E'


def plot_2d_net(nf: NetFold, save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('#F8F9FA')
    ax.axis('off')

    # Draw triangles
    for tri in nf.triangles:
        is_root = (tri.id == nf.root.triangle_id)
        color   = C_FACE_ROOT if is_root else C_FACE
        patch   = plt.Polygon(tri.vertices, closed=True,
                              facecolor=color,
                              edgecolor='#DEE2E6',
                              linewidth=0.5, zorder=1)
        ax.add_patch(patch)
        cx, cy = tri.centroid()
        ax.text(cx, cy, str(tri.id), fontsize=6,
                ha='center', va='center',
                color='#6C757D', zorder=3)

    # Draw fold edges
    for fe in nf.fold_edges:
        tri_a = nf.triangles[fe.tri_a]
        va0   = tri_a.vertices[fe.local_a[0]]
        va1   = tri_a.vertices[fe.local_a[1]]

        is_flat = abs(fe.dihedral_angle - np.pi) < 0.01
        color   = C_FOLD_FLAT if is_flat else C_FOLD_90
        lw      = 0.8 if is_flat else 2.0
        ls      = '--' if is_flat else '-'

        ax.plot([va0[0], va1[0]], [va0[1], va1[1]],
                color=color, linewidth=lw,
                linestyle=ls, zorder=2)

        if not is_flat:
            mid       = (va0 + va1) / 2
            angle_deg = round(np.degrees(fe.dihedral_angle), 1)
            ax.text(mid[0] + 0.04, mid[1] + 0.04,
                    f'{angle_deg}°', fontsize=6,
                    color=C_FOLD_90, zorder=4,
                    fontweight='bold')

    # Draw stitch edges as dashed red arrows between centroids
    for se in nf.stitch_edges:
        ca = nf.triangles[se.tri_a].centroid()
        cb = nf.triangles[se.tri_b].centroid()
        ax.annotate('', xy=cb, xytext=ca,
                    arrowprops=dict(
                        arrowstyle='->',
                        color=C_STITCH,
                        lw=1.2,
                        linestyle='dashed',
                        connectionstyle='arc3,rad=0.3'),
                    zorder=5)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_FACE_ROOT,
                       edgecolor='#DEE2E6',
                       label='Root face (anchor)'),
        mpatches.Patch(facecolor=C_FACE,
                       edgecolor='#DEE2E6',
                       label='Face'),
        plt.Line2D([0],[0], color=C_FOLD_90, lw=2,
                   label='Fold edge (90°)'),
        plt.Line2D([0],[0], color=C_FOLD_FLAT, lw=1,
                   linestyle='--',
                   label='Within-face edge (180°)'),
        plt.Line2D([0],[0], color=C_STITCH, lw=1.5,
                   linestyle='--',
                   label='Stitch edge (closes box)'),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              fontsize=8, framealpha=0.9, fancybox=True)

    ax.set_title('NetFold — 2D Net\nUnit Cube',
                 fontsize=13, fontweight='bold',
                 color=C_LABEL, pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_3d_reconstruction(placed: dict, nf: NetFold,
                           save_path: str = None):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F0F4F8')
    fig.patch.set_facecolor('#F0F4F8')

    faces_3d = []
    colors   = []
    for tri_id, verts_3d in placed.items():
        faces_3d.append(verts_3d)
        is_root = (tri_id == nf.root.triangle_id)
        colors.append('#FFF3CD' if is_root else C_3D_FACE)

    poly = Poly3DCollection(faces_3d, alpha=0.75, zorder=2)
    poly.set_facecolor(colors)
    poly.set_edgecolor(C_3D_EDGE)
    poly.set_linewidth(0.8)
    ax.add_collection3d(poly)

    all_verts = np.vstack(list(placed.values()))
    mn = all_verts.min() - 0.1
    mx = all_verts.max() + 0.1
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_zlim(mn, mx)

    ax.set_xlabel('X', labelpad=8)
    ax.set_ylabel('Y', labelpad=8)
    ax.set_zlabel('Z', labelpad=8)
    ax.set_title('NetFold — 3D Reconstruction\nUnit Cube',
                 fontsize=13, fontweight='bold',
                 color=C_LABEL, pad=15)
    ax.view_init(elev=22, azim=45)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig