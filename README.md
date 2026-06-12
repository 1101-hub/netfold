# NetFold

**A geometry encoding format for folded, fabricated, and simulated surfaces.**

NetFold encodes any closed triangulated 3D surface as a flat 2D net — the unfolded pattern needed to physically cut and fold it from sheet material. Every fold angle, stitch constraint, and island boundary is stored in a single compact file.

---

## What it does

```
OBJ mesh  ──►  encode_mesh()  ──►  .netfold / .nfb  ──►  reconstruct()  ──►  3D mesh
                                      ↓
                                  plot_2d_net()
                                  (fabrication pattern)
```

1. **Encodes** a 3D mesh (OBJ) → 2D flat net via BFS unfolding
2. **Detects overlaps** using SAT (Separating Axis Theorem) per triangle pair
3. **Cuts** overlapping edges, creating multiple non-overlapping **islands**
4. **Serialises** to human-readable JSON (`.netfold`) or compact binary (`.nfb`)
5. **Reconstructs** exact 3D geometry from the flat net + fold angles
6. **Verifies** reconstruction via stitch-edge closure error

---

## Applications

| Domain | Use case |
|---|---|
| **Aero engineering** | Sheet metal panels, fuselage skin unfolding |
| **Formula Student** | Nose cone, bodywork, sidepod fabrication patterns |
| **CFD pre-processing** | Compact mesh exchange format |
| **Simulation** | Geometry exchange between solvers |
| **Manufacturing** | CNC cutting patterns from 3D CAD |

---

## Quick start

```python
from core import encode_mesh, encode_mesh_multi_island, reconstruct, verify_closure
from core import save_netfold, load_netfold
from core.visualise import plot_2d_net, plot_3d_reconstruction

# Encode
nf = encode_mesh_multi_island("my_part.obj", name="nose_cone")
nf.summary()

# Visualise flat net
plot_2d_net(nf, save_path="net.png")

# Save
save_netfold(nf, "nose_cone.netfold")   # JSON
save_netfold(nf, "nose_cone.nfb")       # Binary (~3× smaller)

# Load and reconstruct
nf2 = load_netfold("nose_cone.netfold")
placed = reconstruct(nf2)

# Verify closure
results = verify_closure(nf2, placed)
max_err = max(r['error'] for r in results)
print(f"Max stitch error: {max_err:.2e}")
```

---

## File formats

### `.netfold` — JSON (human-readable)

```json
{
  "version": 1,
  "name": "nose_cone",
  "root": { "triangle_id": 0, "position_3d": [...], "normal_3d": [...] },
  "triangles": [{"id": 0, "vertices": [[x,y],[x,y],[x,y]]}, ...],
  "fold_edges": [{"tri_a": 0, "tri_b": 1, "dihedral_angle": 1.5708, ...}],
  "stitch_edges": [...],
  "cut_edges": [...],
  "islands": [[0,1,2,...], [45,46,...]]
}
```

### `.nfb` — Binary (numpy npz, compressed)

- Same data as JSON, stored as packed float64 arrays
- Bit-exact float round-trip (no `float→str→float` precision loss)
- ~3–10× smaller than JSON depending on mesh size

---

## Key concepts

### Fold edges
Interior edges in the spanning tree. Each stores a **dihedral angle** (radians) and **fold direction** (+1/-1). These are the fold lines on the fabrication pattern.

### Stitch edges
Edges not in the spanning tree — they must **meet in 3D** but are disconnected in the 2D net. Like the last fold that closes a box.

### Cut edges *(multi-island)*
Edges severed to prevent 2D overlap on non-convex shapes. The two triangles belong to separate **islands** in the 2D layout but must join during 3D assembly.

### Islands
Non-overlapping connected regions in the 2D net. On convex shapes (cube, sphere), typically 1 island. On complex non-convex geometry (car body, nose cone), multiple islands.

---

## Format stats (demo meshes)

| Mesh | Triangles | Islands | OBJ | JSON | Binary |
|---|---|---|---|---|---|
| Cube | 12 | 1 | 0.4 KB | 3.4 KB | **1.8 KB** |
| Icosahedron | 20 | 1 | 0.6 KB | 5.9 KB | **2.2 KB** |
| Geodesic sphere n=2 | 320 | 1 | 12.9 KB | 93.9 KB | **12.4 KB** ← *96% of OBJ* |
| FS Nose cone | 720 | 1+ | 21.0 KB | 213.7 KB | **27.8 KB** |

> Binary format is comparable to raw OBJ size while encoding **full fold angle and stitch topology**.
> JSON is human-readable and inspectable with any text editor.

---

## Architecture

```
core/
  structure.py       — NetFold, Triangle2D, FoldEdge, StitchEdge, CutEdge, RootAnchor
  auto_encoder.py    — OBJ → NetFold (encode_mesh, encode_mesh_multi_island)
  reconstruct.py     — NetFold → 3D (reconstruct, verify_closure, refine_closure)
  serialise.py       — JSON + binary save/load
  visualise.py       — 2D net + 3D reconstruction plots
  overlap.py         — SAT overlap detection
  optimal_root.py    — BFS root selection

test/
  test_roundtrip.py  — 27 tests: roundtrip, serialisation, normalisation, multi-island
  test_overlap.py    — SAT unit tests + mesh overlap checks
  conftest.py        — shared fixtures

demo.py              — end-to-end demo with visualisations
gen_nosecone.py      — parametric FS nose cone generator
```

---

## Running tests

```bash
python -m pytest test/ -v
# 27 passed in ~3.5s
```

---

## Roadmap

- [ ] Interactve web viewer (Three.js)
- [ ] Support for open meshes (with boundary)
- [ ] Optimal island layout (minimise bounding box area)
- [ ] Export to SVG / DXF for CNC cutting
- [ ] Python package (`pip install netfold`)

---

## Author

Amulya — built as part of an aero engineering project, extended to a general-purpose fabrication and simulation geometry format.
