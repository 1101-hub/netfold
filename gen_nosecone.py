"""
Generate a parametric Formula Student nose cone .obj
Geometry: Ogive/conical nose cone, closed at the base.
Typical FS nose: ~600mm long, 200mm base radius, ~3.1 fineness ratio.

This produces a water-tight (manifold, closed) triangulated mesh
suitable for NetFold encoding.

Run: python gen_nosecone.py
Output: test/nosecone.obj
"""

import numpy as np
import os

def ogive_radius(x, L, R):
    """Tangent ogive radius at axial position x (0=tip, L=base)."""
    rho = (R**2 + L**2) / (2 * R)
    return np.sqrt(rho**2 - (L - x)**2) - (rho - R)

def generate_nosecone_obj(
    length=0.600,       # m — FS nose cone length
    base_radius=0.200,  # m — base radius
    n_sections=18,      # axial slices (more = smoother but more triangles)
    n_circ=20,          # circumferential points
    out_path=None,
):
    """
    Generate a closed ogive nose cone as a triangulated .obj.
    Returns (vertices, faces) and writes to out_path if given.
    """
    verts = []
    faces = []

    # Tip vertex
    verts.append((0.0, 0.0, 0.0))   # index 0
    tip_idx = 0

    # Section rings from near-tip to base
    xs = np.linspace(length / n_sections, length, n_sections)
    radii = [ogive_radius(x, length, base_radius) for x in xs]

    ring_start = []   # first vertex index of each ring
    for i, (x, r) in enumerate(zip(xs, radii)):
        ring_start.append(len(verts))
        angles = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
        for a in angles:
            verts.append((x, r * np.cos(a), r * np.sin(a)))

    # Cap vertex at base centre
    base_x = length
    verts.append((base_x, 0.0, 0.0))
    base_idx = len(verts) - 1

    n = n_circ

    def ring_idx(ring, j):
        return ring_start[ring] + (j % n)

    # Tip fan — triangles from tip to first ring
    for j in range(n):
        faces.append((tip_idx, ring_idx(0, j+1), ring_idx(0, j)))

    # Side strips between consecutive rings
    for i in range(n_sections - 1):
        for j in range(n):
            a = ring_idx(i,   j)
            b = ring_idx(i,   j+1)
            c = ring_idx(i+1, j+1)
            d = ring_idx(i+1, j)
            # Two triangles per quad
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Base cap fan — triangles from last ring to base centre
    last = n_sections - 1
    for j in range(n):
        faces.append((base_idx, ring_idx(last, j), ring_idx(last, j+1)))

    # Write OBJ
    if out_path:
        with open(out_path, 'w') as f:
            f.write("# Formula Student Ogive Nose Cone\n")
            f.write(f"# Sections: {n_sections}, Circumference points: {n_circ}\n")
            f.write(f"# Length: {length*1000:.0f}mm, Base radius: {base_radius*1000:.0f}mm\n")
            f.write(f"# Triangles: {len(faces)}\n\n")
            f.write("o nosecone\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"  Written {len(verts)} vertices, {len(faces)} triangles -> {out_path}")

    return verts, faces


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "test", "nosecone.obj")
    generate_nosecone_obj(out_path=out)
