# test/gen_geodesic.py
"""
Generates a geodesic sphere OBJ by subdividing an icosahedron n times.
n=1 → 80 faces, n=2 → 320, n=3 → 1280, n=4 → 5120
"""
import numpy as np
from pathlib import Path

PHI = (1 + 5**0.5) / 2

def _normalize(v): return v / np.linalg.norm(v)

def subdivide(verts, faces):
    edge_mid = {}
    new_faces = []
    verts = list(verts)

    def midpoint(i, j):
        key = (min(i,j), max(i,j))
        if key not in edge_mid:
            m = _normalize((verts[i] + verts[j]) / 2)
            edge_mid[key] = len(verts)
            verts.append(m)
        return edge_mid[key]

    for a, b, c in faces:
        ab, bc, ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
        new_faces += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
    return np.array(verts), new_faces

def geodesic_sphere(n_subdivisions=3):
    v = [_normalize(np.array(p)) for p in [
        [-1, PHI, 0],[1,PHI,0],[-1,-PHI,0],[1,-PHI,0],
        [0,-1,PHI],[0,1,PHI],[0,-1,-PHI],[0,1,-PHI],
        [PHI,0,-1],[PHI,0,1],[-PHI,0,-1],[-PHI,0,1]
    ]]
    f = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
         (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
         (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
         (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    verts = np.array(v)
    for _ in range(n_subdivisions):
        verts, f = subdivide(verts, f)
    return verts, f

def write_obj(path, verts, faces):
    with open(path, 'w') as fh:
        for x,y,z in verts:
            fh.write(f"v {x} {y} {z}\n")
        for a,b,c in faces:
            fh.write(f"f {a+1} {b+1} {c+1}\n")

if __name__ == "__main__":
    for n in [1, 2, 3]:
        v, f = geodesic_sphere(n)
        p = Path(__file__).parent / f"geodesic_n{n}.obj"
        write_obj(p, v, f)
        print(f"n={n}: {len(v)} verts, {len(f)} faces → {p}")