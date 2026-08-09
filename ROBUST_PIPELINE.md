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

## Step 3: force, moment and sliding-direction stability

The observed onset of sliding occurred at `beta = 15 deg` while the chute was
also tilted by `alpha = 45 deg`, with continuous floor and wall contact. Under
the current assumption that the PTFE coefficient is equal at both surfaces,
this gives

```text
mu_s = tan(beta) / (sin(alpha) + cos(alpha)) = 0.189469
```

This is an estimate of static friction. The kinetic coefficient during sliding
is not known, so every pose is conservatively evaluated at 11 coefficients
spanning `0 <= mu <= 0.189469`, rather than at one invented exact value.

For each coefficient, a linear program distributes non-negative normal forces
over the boundary vertices of the floor and wall contact regions. A pose passes
when all of the following are true:

- floor and wall forces balance the Y and Z components of gravity;
- the contact forces produce zero net moment about the center of mass;
- the required contact wrench lies strictly inside the available wrench set
  (positive pressure margin, not on a tipping boundary);
- the remaining X acceleration is non-negative, so motion is only downhill.

This is a quasi-static sliding/tipping test, not a simulated drop test. At the
Df1a operating point `alpha = 45 deg`, `beta = 20 deg`, 27 of the 108
theoretical poses pass at every sampled coefficient: all 12 face-face poses,
8 floor-edge/wall-face poses and 7 floor-face/wall-edge poses. Another 20 are
friction-dependent candidates and are retained as uncertain rather than being
silently discarded; 61 fail at every sampled coefficient.

```powershell
uv run --extra dev chute-pose stability "Werkstücke_STL_grob/Df1a.STL" `
  --alpha 45 --beta 20 --onset-alpha 45 --onset-beta 15 `
  --render-output-dir "Poses_Found_Robust/Df1a_stable"
```

The reported pressure margin is dimensionless. Zero means a transition/tipping
boundary; larger positive values mean more reserve inside this particular
contact-wrench model. It is useful for ranking poses, but is not yet an
experimentally calibrated probability of occurrence.

## Discrete part symmetry and physical pose classes

Catalog rotations are representations in the part coordinate system. If a
part has a finite proper rotational symmetry group, rotations `R` and `R @ S`
describe the same occupied geometry for every symmetry element `S`. They must
therefore be reported as one physical pose class.

Symmetry handling is deliberately two-stage:

1. Principal-axis rotations are tested quickly against the complete STL vertex
   set using a reported practical distance tolerance.
2. A matching STEP file is loaded with OpenCascade and the candidate rotation
   is checked using the Boolean symmetric-volume difference of the exact B-Rep.

The STEP check distinguishes exact CAD symmetry from a process-level practical
symmetry. Exact STEP-confirmed symmetries are merged automatically. A merely
practical symmetry is only merged when an explicit, part-specific tolerance is
provided; otherwise the STL candidate is reported but left unmerged. The STEP
support is optional so the normal STL pipeline does not require OpenCascade:

```powershell
uv run --extra step chute-pose symmetry "Werkstücke_STL_grob/Df1a.STL" `
  --tolerance-mm 0.5
```

For Df1a this finds a `C3` candidate. STEP does not confirm it as exact because
the CAD model itself is incorrect: the two rotated sectors have about `0.4038%`
symmetric-volume difference and roughly `0.32 mm` vertex deviation. It is not
merged by default. With the explicitly approved `0.5 mm` tolerance, the
experimentally indistinguishable face-face representations are grouped:

```text
(9, 12, 32)   (24, 26, 28)   (60, 61, 86)   (35, 105, 106)
```

For Ql1i the same automatic procedure finds `C4`; STEP confirms the fourfold
symmetry exactly, and its 24 theoretical rotations reduce to 6 physical pose
classes of four representations each.

## Additional validation parts

At `alpha = 45 deg`, `beta = 20 deg` and the Df1a-derived sampled friction
range, Dl1a has exact STEP-confirmed `C3` symmetry. Its 198 theoretical
representations contain 42 quasi-statically admissible results, which reduce to
14 physical classes: 6 face-face, 4 floor-edge/wall-face and 4
floor-face/wall-edge classes.

Qk1a has no non-trivial full-part rotation symmetry (`C1`). Its faceted convex
hull currently produces 86 support planes and 2232 theoretical orientations.
The quasi-static filter retains 48: 32 face-face and 8 of each face-edge
direction. The large unfiltered count and several visually near-identical
results show that curved/faceted support regions need a separate consolidation
step before Qk1a should be treated as a final pose prediction.

Edge/face configurations that rapidly switch between nearby orientations are
kept separate from symmetry equivalence. They are transition or metastability
candidates and require a later dynamic/contact-mode criterion; geometric
similarity alone must not merge them into a stable pose.

Consequently, the 27 results of the current force/moment filter should be read
as *quasi-statically admissible*, not automatically as experimentally stable.
For Df1a the 12 face-face representations (four physical classes) are accepted
by observation. The remaining edge/face results stay flagged as transition or
metastability candidates until that additional criterion is implemented.

## Disturbance robustness: finite rocking barriers

The nominal wrench test does not represent a part catching on dirt, an edge or
another local irregularity. The first disturbance model therefore computes two
diagnostic first-unloading capacities for every nominally admissible pose:

- the smallest additional braking force per unit mass in `-X`, applied at the
  worst boundary point of either the floor or wall contact;
- the smallest positive or negative pure upset moment about any of the three
  mass-principal part axes.

For every disturbance direction, a linear program redistributes non-negative
floor and wall pressure while preserving force and moment equilibrium. These
capacities are useful diagnostics, but unloading a single contact point is not
necessarily overturning. Qk1a poses 518 and 519 can unload one contact, rock
into another contact and return to the same diagonal pose.

The robust decision therefore adds a finite rocking-energy calculation. For
each pose it samples signed spatial rocking axes and follows the orientation in
small steps out to 5 degrees. At every step the part is translated back against
both chute planes. The weakest peak increase in potential energy is reported
as the equivalent vertical centre-of-mass lift in millimetres.

The first explicitly calibrated decision rule is:

```text
finite rocking barrier >= 0.20 mm
for face-face poses additionally: critical braking reserve >= 0.10 g
```

The 0.20 mm value is a provisional robustness scale for the currently observed
irregularities, not a material constant. It must be validated on further parts.
The conditional face-face braking check remains because a pure face-face pose
does not have the harmless early edge-unloading mechanism of diagonal Qk1a
poses. This rule reproduces all currently supplied labels:

- Df1a: the 27 nominal representations reduce to exactly the 12 observed
  face-face representations, or 4 practical C3 classes;
- Dl1a: the 42 nominal representations reduce to exactly the 4 observed
  classes `15/63/154`, `16/64/153`, `31/109/168`, `34/87/169`;
- the mirrored Dl1a longitudinal classes have only about 0.011 mm barrier,
  while the observed classes have about 0.248 mm;
- Qk1a: only the 16 diagonal edge/face representations remain. Their weakest
  barriers are about 0.339--0.365 mm; the previously predicted face-face
  classes remain below about 0.14 mm.

Run the filter and render the surviving physical classes with:

```powershell
uv run chute-pose disturbance "Werkstücke_STL_grob/Dl1a.STL" `
  --symmetry-tolerance-mm 0.4 `
  --render-output-dir "Poses_Found_Robust/Dl1a_disturbance_robust"
```

## Practical contact-pose equivalence

This is separate from exact STEP part symmetry. Candidate orientations are
complete-link clustered only when every pair stays inside configured angular
and occupied-surface displacement tolerances. An explicitly approved practical
rotation group may also be supplied.

For Qk1a, poses 130, 525, 528 and 1102 differ by quarter turns about the part Z
axis. The full STEP part is not exactly C4, but all four placements differ by at
most about `0.523 mm`; the experimentally approved functional equivalence is
therefore represented by an explicit `0.6 mm` practical C4 tolerance. After
finite-disturbance filtering, Qk1a retains four practical classes (16 catalog
representations) rather than the 48 nominal representations. Contact clustering
uses at least the explicitly approved symmetry tolerance, so the measured
0.523 mm C4 deviation is not accidentally compared against the smaller 0.5 mm
default. Its practical 1 degree angular tolerance also consolidates support-
facet variants which differ by about 0.864 degrees and only 0.304 mm occupied-
surface displacement.
