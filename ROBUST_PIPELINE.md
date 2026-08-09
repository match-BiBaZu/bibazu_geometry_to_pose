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

