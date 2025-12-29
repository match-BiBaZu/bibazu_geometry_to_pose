# Geometric Resting‑Pose Finder


Determine every *geometrically possible* resting pose of a rigid 3‑D part on a slide such that

* the part **rests on one of its convex‑hull faces** (bottom plane) and
* **exactly three vertices** touch the opposite slide surface (top plane).

Cylindrical features from the STEP model are used to discretise twist about axes.


The scripts convert common mesh formats, extract a convex hull, search the orientation space, prune unstable or duplicate solutions and finally export a CSV + PNG visualisations for each unique pose. The output plots should look like as the examples for Teil 5, shown below:
<img width="5772" height="2464" alt="Teil_5_poses_on_face_0" src="https://github.com/user-attachments/assets/5b26c106-30a4-4be3-8a11-243ae30976df" />
<img width="5772" height="2464" alt="Teil_5_poses_on_face_1" src="https://github.com/user-attachments/assets/1ba06ef8-3b82-4db5-8d44-60ab6c8f2eb2" />
<img width="5772" height="2464" alt="Teil_5_poses_on_face_2" src="https://github.com/user-attachments/assets/b4e0bfce-d3f6-4216-b433-18424c3851d6" />
<img width="5772" height="2464" alt="Teil_5_poses_on_face_4" src="https://github.com/user-attachments/assets/032bf715-239f-43a1-bf71-1aee9402094d" />
<img width="5772" height="2464" alt="Teil_5_poses_on_face_5" src="https://github.com/user-attachments/assets/6cadbb24-ad20-4e30-a42a-effcc09b2b9b" />

---

## TL;DR

```bash
# Python ≥3.10
pip install -U numpy scipy trimesh matplotlib  # core deps
# optional (only if you use the Blender STL→OBJ step)
pip install -U bpy

python Main.py  # runs the full pipeline on the parts listed inside Main.py
```

Outputs land in `csv_outputs/` (CSV) and `Poses_Found/` (PNGs).

---

## What this repository does

1. **Convert & prepare meshes** (STL→OBJ, optional) and build a **convex hull** of the part.
2. **Enumerate candidate orientations** by aligning each hull face to the slide’s bottom plane and sweeping the **shadow edge** around +Y.
3. **Attach cylinder/round‑edge axes** parsed from the STEP file to every pose.
4. **Cull duplicates & unstable poses** (centre of mass inside the support polygon on the base).
5. **Discretise twist** about detected cylinder axes and **export** quaternions + visualisations.

---

## File map

| File                           | Purpose                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `Main.py`                      | Orchestrates the whole pipeline (per‑part loop, I/O paths, parameters).                                                                     |
| `obj_convex_hull_extractor.py` | Builds a convex hull mesh from an OBJ and saves it as OBJ.                                                                                  |
| `step_find_all_cylinders.py`   | Parses the STEP file and writes CSV with **all** cylinders / circular edges and their axes.                                                 |
| `PoseFinder.py`                | Generates candidate orientations by **face‑normal** alignment and **shadow‑edge** alignment; loads cylinder axes from CSV; writes pose CSV. |
| `PoseEliminator.py`            | Removes duplicates, filters **unstable** poses, and **discretises** rotation about cylinder axes (with wobble tolerance).                   |
| `PoseVisualizer.py`            | Renders the part, reference planes, centroid, shadow, and (optionally) cylinder axes; saves per‑face PNGs.                                  |
| `obj_to_ply_converter.py`      | Convenience: OBJ → PLY via `trimesh`.                                                                                                       |

> Note: `Main.py` calls `stl_to_obj_converter.py`. If you don’t use that step (or don’t have the script), provide **OBJ** files directly and comment that line.

---

## Inputs / expected layout

```
repo/
 ├─ Main.py
 ├─ Werkstücke_STL_grob/               # per‑part geometry
 │   ├─ <PartName>.STEP                # STEP model (for cylinder/edge axes)
 │   ├─ <PartName>.STL                 # raw mesh (optional if you already have OBJ)
 │   └─ <PartName>.obj                 # triangulated mesh used by the pipeline
 ├─ csv_outputs/                        # created automatically (axes + poses CSV)
 └─ Poses_Found/                        # created automatically (visualisations)
```

`Main.py` contains the list of `workpiece_names` you want to process and a small heuristic to set **centering mode** per part (see below).

---

## How it works (brief)

* **Convex hull & centering.** Both the part and hull are translated so the **hull centroid** is at the origin. Cylinder axes from the CSV are shifted by the same offset.
* **Face alignment.** For each unique outward hull normal, rotate so that normal aligns with the slide’s **−Z** direction. (This yields the base‑contact candidates.)
* **Shadow sweep.** For each face‑aligned pose, rotate around **Z** so the chosen 2‑D hull **edge** (from the left‑bottom shadow vertex) aligns with **+Y**; then step through the polygon’s internal angles to enumerate distinct edge contacts.
* **Stability test.** Project the rotated mesh’s vertices that lie on the lowest **Z** to the plane, build their 2‑D convex hull (support polygon), and keep poses where the **centre of mass** projects **inside**.
* **Cylinder discretisation.** Group poses by cylinder‑axis direction, classify cylinder vs. non‑cylinder contacts on base/back planes, then **downsample** twist along each axis according to `rotation_steps` with a `wobble_angle` tolerance.

---

## Configuration knobs (edit in code)

* `PoseFinder(..., tolerance=1e-5, is_workpiece_centered=<0|1|2>)`

  * **tolerance**: numeric tolerance for equality/rounding.
  * **centering mode**: `0` = use hull centroid; `1/2` = special‑case offsets for specific parts.
* `PoseEliminator(..., rotation_steps=<int>, wobble_angle=<deg>)`

  * `rotation_steps`: how many **discrete** twists to keep around each cylinder axis.
  * `wobble_angle`: allowable axis misalignment (converted to a distance band).

`Main.py` contains a heuristic: parts with names starting with **K** (circular) receive centering mode `2`; `Rl1a` uses mode `1`; otherwise `0`.

---

## Running the pipeline

1. Ensure each part has **STEP** and **STL/OBJ** present in `Werkstücke_STL_grob/`.
2. (Optional) If using STL, keep the STL→OBJ call; otherwise provide an OBJ and comment that line in `Main.py`.
3. Run `python Main.py`.
4. Inspect:

   * `csv_outputs/<Part>_cylinder_properties.csv` (axes parsed from STEP)
   * `csv_outputs/<Part>_candidate_rotations.csv` (PoseID, FaceID, EdgeID, QuatX/Y/Z/W)
   * `Poses_Found/<Part>_poses_on_face_*.png` (per‑face grids of unique poses)

---

## Module requirements

* **numpy**, **scipy**, **trimesh**, **matplotlib**
  Install latest stable releases: `pip install -U numpy scipy trimesh matplotlib`
* **bpy** *(optional)* – only if you use the Blender‑based STL→OBJ conversion: `pip install -U bpy` or run the converter **inside Blender**.

---

## Notes & limitations

* The search operates on the **convex hull**; fine non‑convex features won’t create extra resting poses.
* Stability uses a **quasi‑static COM‑inside‑support** test; friction and micro‑wobble are approximated by `wobble_angle`.
* STEP parsing is lightweight text parsing; unusual STEP flavours may need adjustments.

---

## Minimal snippet

```python
from PoseFinder import PoseFinder
from PoseEliminator import PoseEliminator

pf = PoseFinder('part_convex_hull.obj','part.obj',1e-5,0)
rots, shadows, cyl = pf.find_candidate_rotations_by_face_and_shadow_alignment()
pe = PoseEliminator('part_convex_hull.obj','part.obj',0.01,12,15)
rots, shadows, cyl = pe.remove_duplicates(rots, shadows, cyl)
rots, shadows, cyl = pe.remove_unstable_poses(rots, shadows, cyl)
rots, shadows, cyl = pe.discretise_rotations(rots, shadows, cyl)
pf.write_candidate_rotations_to_file(rots, 'csv_outputs/part_candidate_rotations.csv')
```

## MIT License
