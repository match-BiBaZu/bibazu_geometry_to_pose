# bibazu_geometry_to_pose

**Purpose – one sentence**
Determine every *geometrically possible* resting pose of a rigid 3‑D part on a slide such that

* the part **rests on one of its convex‑hull faces** (bottom plane) and
* **exactly three vertices** touch the opposite slide surface (top plane).

The scripts convert common mesh formats, extract a convex hull, search the orientation space, prune unstable or duplicate solutions and finally export a CSV + PNG visualisations for each unique pose.

---

## Quick start

```bash
# 1 Install Python ≥3.10 and Blender ≥3.6 (for *.stl → *.obj* conversion).
# 2 Install core Python libraries
pip install numpy>=2.0 scipy>=1.14 trimesh>=4.2 matplotlib>=4.1 networkx>=3.5

# Optional – if you want to run the STL converter outside Blender:
pip install bpy>=3.6

# 3 Clone repo and run
python Main.py            # processes all workpieces listed in *Main.py*
```

Outputs appear next to each workpiece:

* `<name>_candidate_rotations.csv` – quaternion for every valid pose.
* `Poses_Found/<name>_poses_on_face_*.png` – gallery of poses per resting face.

---

## Folder / script map

| File                           | Responsibility                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `Main.py`                      | High‑level pipeline: conversion → convex hull → pose search → filtering → visualisation                            |
| `stl_to_obj_converter*.py`     | Uses **Blender** to convert / simplify STL and (optionally) replace circular arcs with octagons                    |
| `obj_convex_hull_extractor.py` | Extract convex hull with **SciPy** + **trimesh**                                                                   |
| `PoseFinder.py`                | 1) align every hull face to −Z 2) spin part around Z to place a hull edge along +Y 3) output candidate quaternions |
| `PoseEliminator.py`            | Remove duplicate & unstable poses (centre‑of‑mass inside support polygon)                                          |
| `PoseVisualizer.py`            | Plot mesh, projected shadow & reference planes, save figure                                                        |
| `obj_to_ply_converter.py`      | Utility if PLY export is required                                                                                  |

---

## Algorithm in brief

1. **Mesh preparation** – simplify & scale STL → OBJ → convex hull (guarantees convex bottom/top contact).
2. **Face alignment** – rotate every outward hull face normal to the slide normal (‑Z).
3. **Edge alignment** – spin around Z so the shadow edge that starts at the left‑bottom vertex aligns with +Y.
4. **Stability test** – keep rotations where the centre of mass projects inside the support polygon on the slide.
5. **Symmetry merge** – cluster equivalent poses within a Euclidean vertex distance ≤ 0.1 × feature size.
6. **Export** – write unique quaternions + images.

---

## Required Python modules (latest stable 2025‑07‑18)

| Package              | Reason                                | Install                                   |
| -------------------- | ------------------------------------- | ----------------------------------------- |
| **numpy ≥ 2.0**      | maths & array ops                     | `pip install numpy`                       |
| **scipy ≥ 1.14**     | convex hull, rotations                | `pip install scipy`                       |
| **trimesh ≥ 4.2**    | mesh IO & geometry                    | `pip install trimesh`                     |
| **matplotlib ≥ 4.1** | 3‑D & 2‑D plots                       | `pip install matplotlib`                  |
| **bpy ≥ 3.6**        | Blender API for STL→OBJ               | `pip install bpy` *or* run inside Blender |
| **tkinter**          | screen size (std‑lib on most distros) | –                                         |
| **networkx ≥ 3.5**   | graph algorithms (symmetry merge)      | `pip install networkx`                    |

> **Tip**  If running on a headless server you can skip the STL step by providing OBJ files directly.

---

## Minimal example

```python
from PoseFinder import PoseFinder
from PoseEliminator import PoseEliminator
from pathlib import Path

mesh = Path('sample.obj')
hull = Path('sample_convex_hull.obj')

pf = PoseFinder(hull, mesh, 1e‑5)
rots, shadows = pf.find_candidate_rotations_by_face_and_shadow_alignment()
pe = PoseEliminator(hull, mesh, 1e‑2)
unique, shadows = pe.remove_duplicates(rots, shadows)
stable, shadows = pe.remove_unstable_poses(unique, shadows)
print(f"{len(stable)} stable poses")
```

---

## MIT License
