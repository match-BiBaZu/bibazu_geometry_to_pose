# Robust chute-pose pipeline

This package is the clean implementation path for predicting poses in the
doubly inclined BiBaZu chute. The legacy scripts remain available while this
pipeline is developed and verified step by step.

## Step 1: coordinate and geometry contract

The chute-fixed coordinate system is right-handed:

- `+X`: downhill along the chute
- `+Y`: away from the side wall; the admissible part region is `y >= 0`
- `+Z`: away from the floor; the admissible part region is `z >= 0`
- floor: `z = 0`
- wall: `y = 0`
- common floor/wall seam: `(x, 0, 0)`, parallel to X

Only simultaneous floor-and-wall poses are part of the pose catalog.

Starting from the neutral position on the table, the chute is rotated first by
`beta` about the original Y axis. It is then rotated by `alpha` about the moved,
chute-fixed X axis:

```text
R_world_from_chute = R_y(beta) @ R_x(alpha)
```

For the first Df1a test, `alpha = 45 deg` and `beta = 20 deg`. Expressed in the
chute frame, gravity then has positive X, negative Y and negative Z components:

```text
g_chute = ( 3.355218, -6.518383, -6.518383 ) m/s^2
```

This drives a part downhill in `+X`, toward the wall in `-Y`, and toward the
floor in `-Z`.

Input geometry is measured in millimetres. A homogeneous mass distribution is
assumed. Meshes must be watertight, consistently wound positive-volume solids;
the validator deliberately does not repair invalid geometry silently.

## Run the Step-1 Df1a inspection

From this repository:

```powershell
uv run --extra dev chute-pose inspect "Werkstücke_STL_grob/Df1a.STL" --alpha 45 --beta 20
uv run --extra dev pytest
```

The command does not write result files. Add `--json` for machine-readable
output.

## Step 2: theoretical simultaneous-contact catalog

Every maximal coplanar polygon on the convex hull is retained as a possible
support face, including small sloped faces and chamfers. Candidate orientations
are generated in both directions:

- a convex-hull face on the floor plus an edge or face at the wall;
- an edge or face on the floor plus a convex-hull face at the wall.

Pure point contacts are pose transitions and are excluded. Edge-edge contact
has a remaining rotational degree of freedom, so it is not an isolated,
perturbation-resistant pose and is also excluded from the pose catalog. No
alpha/beta stability decision is made in this step.

```powershell
uv run --extra dev chute-pose catalog "Werkstücke_STL_grob/Df1a.STL"
```

With the locked dependencies and tolerances, the Df1a baseline contains 9
convex support faces and 108 unfiltered theoretical poses:

- 48 floor-edge / wall-face poses;
- 48 floor-face / wall-edge poses;
- 12 floor-face / wall-face poses.

These counts are regression-tested. They are not a prediction that all 108
orientations are stable at `alpha = 45 deg`, `beta = 20 deg`; that filtering is
the purpose of Step 3.

Render all unfiltered candidates as grouped technical contact sheets:

```powershell
uv run --extra dev chute-pose render "Werkstücke_STL_grob/Df1a.STL" `
  --output-dir "Poses_Found_Robust/Df1a_theoretical"
```

The plots show floor contacts in green, wall contacts in orange and vertices
touching the common floor-wall seam in red.
