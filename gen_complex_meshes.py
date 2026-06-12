"""
Generate complex test meshes for NetFold stress testing.

Shapes:
  torus.obj         — genus-1 (hole), 720+ triangles
  ellipsoid.obj     — non-uniform curvature, 720 triangles
  sidepod.obj       — FS sidepod body (elongated + curved shoulder)
  blade.obj         — swept aerofoil / turbine blade body

Run: python gen_complex_meshes.py
"""

import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'test')

def write_obj(path, verts, faces, name="mesh"):
    with open(path, 'w') as f:
        f.write(f"# {name}\n")
        f.write(f"# Vertices: {len(verts)}, Faces: {len(faces)}\n\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  {name}: {len(verts)} verts, {len(faces)} tris -> {path}")


# ── 1. TORUS ─────────────────────────────────────────────────────────────────
def gen_torus(R=1.0, r=0.35, n_major=24, n_minor=16):
    """
    Torus: major radius R (centre to tube centre), minor radius r (tube radius).
    n_major × n_minor grid, triangulated.
    """
    verts = []
    for i in range(n_major):
        phi = 2 * np.pi * i / n_major
        for j in range(n_minor):
            theta = 2 * np.pi * j / n_minor
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = (R + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            verts.append((x, y, z))

    faces = []
    for i in range(n_major):
        for j in range(n_minor):
            # Vertex indices with wraparound
            a = i * n_minor + j
            b = i * n_minor + (j + 1) % n_minor
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = ((i + 1) % n_major) * n_minor + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    return verts, faces


# ── 2. ELLIPSOID ──────────────────────────────────────────────────────────────
def gen_ellipsoid(a=1.5, b=1.0, c=0.6, n_lat=20, n_lon=24):
    """
    Ellipsoid with semi-axes a, b, c.
    UV sphere parameterisation, triangulated.
    """
    verts = []
    faces = []

    # Poles
    verts.append((0, 0, c))   # north pole: index 0
    verts.append((0, 0, -c))  # south pole: index 1

    ring_start = 2
    rings = []
    lats = np.linspace(np.pi * 0.05, np.pi * 0.95, n_lat)  # avoid degenerate poles
    for lat in lats:
        rings.append(len(verts))
        for lon_i in range(n_lon):
            lon = 2 * np.pi * lon_i / n_lon
            x = a * np.sin(lat) * np.cos(lon)
            y = b * np.sin(lat) * np.sin(lon)
            z = c * np.cos(lat)
            verts.append((x, y, z))

    def ridx(ring, j):
        return rings[ring] + j % n_lon

    # North cap
    for j in range(n_lon):
        faces.append((0, ridx(0, j), ridx(0, j+1)))

    # Body strips
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a_ = ridx(i, j); b_ = ridx(i, j+1)
            c_ = ridx(i+1, j+1); d_ = ridx(i+1, j)
            faces.append((a_, b_, c_))
            faces.append((a_, c_, d_))

    # South cap
    last = n_lat - 1
    for j in range(n_lon):
        faces.append((1, ridx(last, j+1), ridx(last, j)))

    return verts, faces


# ── 3. FS SIDEPOD ─────────────────────────────────────────────────────────────
def gen_sidepod(length=1.5, width=0.35, height=0.45,
                n_cross=20, n_len=30):
    """
    Formula Student sidepod: roughly an elongated body with rounded cross-section.
    Cross-section: superellipse (squircle) transitioning from inlet to tail.
    Closed at both ends with fan caps.
    """
    def superellipse(t, w, h, n=2.5):
        """Superellipse cross-section: |x/w|^n + |y/h|^n = 1"""
        angles = np.linspace(0, 2*np.pi, n_cross, endpoint=False)
        pts = []
        for a in angles:
            # Parametric superellipse
            ca, sa = np.cos(a), np.sin(a)
            x = w * np.sign(ca) * abs(ca) ** (2/n)
            y = h * np.sign(sa) * abs(sa) ** (2/n)
            pts.append((x, y))
        return pts

    verts = []
    faces = []

    xs = np.linspace(0, length, n_len)

    # Cross-section shape varies along length:
    # inlet (x=0): narrow and tall (inlet mouth)
    # body (x=0.3L): widest
    # tail (x=L): tapered to small oval
    def cross_dims(t):
        """t in [0,1] along length"""
        # Width profile: narrow at inlet, peak at 0.4, taper at tail
        w = width * (0.4 + 1.2 * np.sin(np.pi * t) * (1 - 0.3*t))
        # Height profile: taller at inlet, flatter at tail
        h = height * (0.8 + 0.4 * np.exp(-3*t) - 0.3 * t)
        return w, h

    # Inlet cap centre
    verts.append((0.0, 0.0, height * 0.5))
    inlet_idx = 0

    ring_starts = []
    for xi, x in enumerate(xs):
        t = x / length
        w, h = cross_dims(t)
        ring_starts.append(len(verts))
        pts = superellipse(t, w, h)
        for px, py in pts:
            verts.append((x, px, py + height * 0.1))  # slight z-offset for ground clearance

    # Tail cap centre
    verts.append((length, 0.0, height * 0.1))
    tail_idx = len(verts) - 1

    def ridx(ring, j):
        return ring_starts[ring] + j % n_cross

    # Inlet fan cap
    for j in range(n_cross):
        faces.append((inlet_idx, ridx(0, j+1), ridx(0, j)))

    # Body strips
    for i in range(n_len - 1):
        for j in range(n_cross):
            a = ridx(i, j);   b = ridx(i, j+1)
            c = ridx(i+1, j+1); d = ridx(i+1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Tail fan cap
    last = n_len - 1
    for j in range(n_cross):
        faces.append((tail_idx, ridx(last, j), ridx(last, j+1)))

    return verts, faces


# ── 4. SWEPT AEROFOIL / TURBINE BLADE ────────────────────────────────────────
def gen_blade(chord=0.12, span=0.08, twist_deg=30.0,
              n_span=20, n_profile=24):
    """
    Swept NACA0012 aerofoil cross-section rotated (twisted) along span.
    Closed at root and tip with flat caps.
    This is a representative turbine blade / compressor vane body.
    """
    def naca0012(n=n_profile):
        """NACA0012 profile points, normalised chord 0..1"""
        # Upper surface
        t_vals = np.linspace(0, 1, n // 2 + 1)
        upper = []
        for t in t_vals:
            y = 5 * 0.12 * (0.2969*np.sqrt(t) - 0.1260*t
                             - 0.3516*t**2 + 0.2843*t**3 - 0.1015*t**4)
            upper.append((t, y))
        # Lower surface (reverse, skip endpoints to avoid duplicates)
        lower = [(t, -y) for t, y in reversed(upper[1:-1])]
        return upper + lower  # closed loop

    profile_base = naca0012()
    n_prof = len(profile_base)

    verts = []
    faces = []

    ring_starts = []
    span_vals = np.linspace(0, span, n_span)

    for si, s in enumerate(span_vals):
        twist = np.radians(twist_deg * s / span)
        ring_starts.append(len(verts))
        for (xp, yp) in profile_base:
            # Scale by chord, apply twist rotation, offset by span
            xc = xp * chord - chord * 0.25  # quarter-chord pivot
            yc = yp * chord
            # Twist (in the x-y plane)
            xr = xc * np.cos(twist) - yc * np.sin(twist)
            yr = xc * np.sin(twist) + yc * np.cos(twist)
            verts.append((xr, yr, s))

    # Root cap centre
    verts.append((0.0, 0.0, 0.0))
    root_idx = len(verts) - 1

    # Tip cap centre
    verts.append((0.0, 0.0, span))
    tip_idx = len(verts) - 1

    def ridx(ring, j):
        return ring_starts[ring] + j % n_prof

    # Root fan cap (winding reversed for outward normal)
    for j in range(n_prof):
        faces.append((root_idx, ridx(0, j), ridx(0, j+1)))

    # Body strips
    for i in range(n_span - 1):
        for j in range(n_prof):
            a = ridx(i, j);   b = ridx(i, j+1)
            c = ridx(i+1, j+1); d = ridx(i+1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Tip fan cap
    last = n_span - 1
    for j in range(n_prof):
        faces.append((tip_idx, ridx(last, j+1), ridx(last, j)))

    return verts, faces


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating complex test meshes...\n")

    v, f = gen_torus()
    write_obj(os.path.join(OUT_DIR, "torus.obj"), v, f, "torus")

    v, f = gen_ellipsoid()
    write_obj(os.path.join(OUT_DIR, "ellipsoid.obj"), v, f, "ellipsoid")

    v, f = gen_sidepod()
    write_obj(os.path.join(OUT_DIR, "sidepod.obj"), v, f, "sidepod")

    v, f = gen_blade()
    write_obj(os.path.join(OUT_DIR, "blade.obj"), v, f, "blade")

    print("\nDone. Run: python stress_test.py")
