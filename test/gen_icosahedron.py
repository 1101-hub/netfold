# run once to generate: python test/gen_icosahedron.py
import numpy as np, os

phi = (1 + np.sqrt(5)) / 2
r   = np.sqrt(1 + phi**2)
verts = [
    [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
    [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
    [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
]
verts = [[x/r for x in v] for v in verts]

faces = [
    [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
    [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
    [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
    [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
]

out = os.path.join(os.path.dirname(__file__), "icosahedron.obj")
with open(out, "w") as f:
    for v in verts:
        f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print("written:", out)