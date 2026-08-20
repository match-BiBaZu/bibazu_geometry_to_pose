# BiBaZu geometry-to-pose planner

This repository generates physically admissible chute poses, stability-ranked
pose sheets, and reorientation roadmaps directly from solid STL geometry. STEP
files are retained beside the STL files for exact symmetry and CAD checks.

The supported implementation is the `chute_pose` package in `src/`. The former
OBJ/CSV-based research pipeline and its pose images are preserved under
[`legacy/`](legacy/) for reference only.

## Current defaults

Roadmap and filtered-pose generation now use the following defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| Pose ranking | `rocking` | Lower pose numbers have a larger conservative rocking barrier. |
| Robustness method | `rocking` | Robust/metastable classification uses the finite rocking barrier and the face-face braking safety gate. |
| Friction policy | `zero` | The inferred friction-range barrier is off; nominal admissibility is evaluated at `mu = 0` only. |
| Rocking threshold | `0.20 mm` | Minimum centre-of-mass rise required before a pose is considered robust. |
| Face-face braking threshold | `0.10 g` | Additional safety gate for poses simultaneously supported by floor and wall faces. |
| Roadmap symmetry tolerance | `0.05 mm` | Maximum STL mapping error used for practical rotational-symmetry detection. |

The zero-friction policy does **not** disable force and moment balance. It
removes the inferred static-friction sweep and tests the nominal contact system
once at `mu = 0`. Use `--friction-policy range` only when friction-dependent
diagnostics are intentionally required.

## Setup and launcher

The PowerShell launcher uses the same Python environment as the Reorientation
Control GUI:

```powershell
cd C:\Users\Administrator\Documents\Dashas_ws\bibazu_geometry_to_pose
.\chute-pose.ps1 --help
```

The package can also be invoked from an activated environment:

```powershell
python -m chute_pose.cli --help
```

Active workpiece inputs live in `Werkstücke_STL_grob/`:

- `<part>.STL` is the triangulated solid used by pose and roadmap calculations.
- `<part>.STEP` is the optional CAD reference used to confirm exact symmetry.

OBJ, MTL, precomputed convex-hull OBJ, PLY, and proprietary CAD exports are not
needed by the modern pipeline.

## Coordinate and geometry contract

The chute frame is right-handed: `+X` points downhill, `+Y` points away from
the back wall, and `+Z` points away from the floor. The floor is `z = 0`, the
wall is `y = 0`, and their common seam is parallel to X. All displayed axes use
this same internal calculation frame.

Starting from the neutral chute, beta is applied about the original Y axis and
alpha about the moved, chute-fixed X axis. With active column-vector rotations:

```text
R_world_from_chute = R_y(beta) @ R_x(alpha)
```

At the defaults `alpha=45 deg` and `beta=20 deg`, gravity expressed in chute
coordinates is `(3.355218, -6.518382, -6.518382) m/s^2`. Geometry is interpreted
in millimetres with homogeneous mass density. The modern pipeline validates the
input solid but does not silently repair or rescale it.

## Common commands

Generate symmetry-unique quasi-static pose sheets with the defaults
(rocking-ranked, zero friction):

```powershell
.\chute-pose.ps1 stability .\Werkstücke_STL_grob\Df1a.STL `
  --render-output-dir .\Poses_Found_Robust\Df1a_quasistatic
```

Generate a verified roadmap with the defaults:

```powershell
.\chute-pose.ps1 roadmap .\Werkstücke_STL_grob\Df1a.STL `
  --output-dir .\Poses_Found_Robust\Df1a_roadmap_verified `
  --geometry-status verified
```

The default flags are equivalent to:

```text
--pose-ranking rocking --robustness-method rocking --friction-policy zero
```

Request the former inferred-friction sweep explicitly:

```powershell
.\chute-pose.ps1 stability .\Werkstücke_STL_grob\Df1a.STL `
  --friction-policy range `
  --exhaustive-friction-diagnostics
```

### Other commands

Validate a mesh and inspect its scale, mass centre, hull, and chute gravity:

```powershell
.\chute-pose.ps1 inspect .\Werkstücke_STL_grob\Df1a.STL --json
```

Enumerate or render the complete unfiltered contact-pose catalog:

```powershell
.\chute-pose.ps1 catalog .\Werkstücke_STL_grob\Df1a.STL --json
.\chute-pose.ps1 render .\Werkstücke_STL_grob\Df1a.STL `
  --output-dir .\Poses_Found_Robust\Df1a_theoretical
```

Check STL symmetry and, when available, confirm it against exact STEP geometry:

```powershell
.\chute-pose.ps1 symmetry .\Werkstücke_STL_grob\Df1a.STL `
  --step .\Werkstücke_STL_grob\Df1a.STEP `
  --tolerance-mm 0.05 --json
```

Render the disturbance diagnostics or find a route through an exported roadmap:

```powershell
.\chute-pose.ps1 disturbance .\Werkstücke_STL_grob\Df1a.STL `
  --render-output-dir .\Poses_Found_Robust\Df1a_disturbance

.\chute-pose.ps1 route `
  .\Poses_Found_Robust\Df1a_roadmap_verified\Df1a_roadmap.json `
  --start-pose 0 --target-pose 3
```

Run `.\chute-pose.ps1 <command> --help` for every available option.

## Processing and filter algorithms

The filters are applied in this order.

### 1. Solid-mesh validation

The loader rejects empty, non-finite, non-watertight, or zero-volume meshes.
Mass centre, scale, convex support geometry, and chute coordinates are then
derived from the STL. This prevents invalid geometry from producing plausible
but meaningless poses.

### 2. Contact-pose enumeration

Every maximal coplanar convex-hull support polygon, including chamfers and small
sloped faces, is considered. The planner enumerates simultaneous floor-wall
contact in both directions: a face on the floor with an edge or face at the
wall, and an edge or face on the floor with a face at the wall. The chute angle
and stability model do not filter this theoretical catalog.

Pure point contacts are transitions rather than retained poses. A generic
edge-edge contact has a free rotational degree of freedom and is not treated as
an isolated pose; continuously symmetric rolling parts instead use their
dedicated rolling-contact handling. Circular parts receive a continuous-
symmetry precheck so fine STL facets do not become thousands of artificial
physical poses.

After enumeration, the planner reconstructs contact from the full STL and uses
labels such as `2-point`, `edge`, `edge+point`, and `face`. Highlighted contact
edges are real mesh adjacencies, not lines inferred between unrelated vertices.

### 3. Rotational-symmetry reduction

Orientations related by a rotational symmetry are grouped into one physical
pose class. Symmetry handling is deliberately two-stage: the complete STL
vertex set first supplies a practical mapping error, then a matching STEP file
can confirm the candidate using exact B-Rep geometry. Exact CAD-confirmed
symmetry is merged automatically. Practical-only symmetry requires an explicit,
part-appropriate tolerance. STEP verification uses the optional OpenCascade
support; STL analysis remains usable without it. The exported pose number is
the compact physical class number, not the original catalog ID.

### 4. Zero-friction quasi-static equilibrium (default)

For each contact pose, a linear force-and-moment equilibrium problem distributes
normal contact loads across the separate floor and wall contacts. A pose passes
when equilibrium exists, all required contact loads remain compressive, the
contact-load margin is positive, and motion is compatible with the chute's
positive X convention.

With the default `friction_policy=zero`, this is evaluated only at `mu = 0`.
The optional `range` policy repeats it from zero through the static-friction
coefficient inferred from the configured onset angles and accepts only poses
that pass every sample. That range is deliberately no longer a default filter.

Assuming equal friction at the floor and wall, the inferred upper coefficient
is:

```text
mu_s = tan(onset_beta) / (sin(onset_alpha) + cos(onset_alpha))
```

For the historical onset measurement `onset_alpha=45 deg` and
`onset_beta=15 deg`, this gives `mu_s=0.189469`. It estimates static friction,
not the unknown kinetic coefficient during motion.

The **contact load balance index** is the normalized minimum load margin in
`0..1`. It describes how evenly the equilibrium can stay inside the admissible
contact-load region; it is not a probability.

### 5. Practical contact equivalence

This clustering is separate from exact CAD symmetry. Candidates are grouped by
their pair of contact dimensions, then complete-link clustered only when every
pair remains within the configured angular and occupied-surface displacement
tolerances. The contact-dimensional signature is independent of which chute
plane received each dimension, allowing near-bisector mesh facets to merge when
floor and wall labels exchange; the whole-part orientation must still pass both
geometric tolerances. Conservative class values use the weakest member, so
symmetry or tessellation cannot make a class appear stronger than one of its
representations.

After classification, physical pose IDs are reassigned contiguously from zero
in descending order of the selected conservative class score. The exports keep
the source IDs as `original_catalog_pose_id` and
`equivalent_catalog_pose_ids`; do not persist a bare pose number without its
matching roadmap file because remeshing can change the numbering.

### 6. Finite rocking barrier (default ranking and robustness)

The planner samples signed spatial tipping axes around the active contact
boundary. For each direction it advances the orientation in small seated steps
out to 5 degrees, translating the part back against both chute planes at every
step. The weakest peak potential-energy rise is expressed as the equivalent
vertical centre-of-mass lift: the **rocking barrier**, measured in millimetres.

- Pose classes are numbered from highest to lowest rocking barrier.
- A finite pose is robust when its conservative class barrier is at least
  `0.20 mm`.
- A quasi-static pose below that threshold is metastable.

This is a geometric energy-barrier proxy: larger values require a larger
disturbance to initiate tipping. It is not a probability or a force. The
`0.20 mm` cutoff is a provisional process scale for the observed chute
irregularities, not a material constant, and should be checked against physical
trials for new part families.

### 7. Face-face braking safety gate

A face-face pose can have a large rocking barrier while still releasing too
easily under downhill braking. Such poses must also meet the default critical
braking capacity of `0.10 g`. Edge-contact and rolling cases continue to use
their applicable rocking/continuous-contact logic. This gate supplements the
rocking classifier; it is not the old friction-range pose barrier.

### 8. Roadmap transition construction

The roadmap connects physical pose classes through pure rotations about the
chute-fixed X, Y, and Z axes. Commanded combined-axis rotations are not
automatically synthesized. X actions depend on whether the selected main face
is on the floor or wall and whether it passes the configured intrinsic-size
gate; Y and Z actions allow both signs. Metastable poses may additionally have
zero-actuation `passive_tip` edges found from seated-energy paths about other
axes.

For an actuated transition, the target basin supplies a capture interval. Its
transparent geometric score is:

```text
capture_fraction = capture_width / available_action_angle_span
barrier_score = min(1, target_rocking_barrier / 0.20 mm)
geometric_score = capture_fraction * barrier_score
```

Transition feasibility also uses settling states and contact topology. These
scores are deterministic geometry measures, not empirical success
probabilities. The `0.20 mm` rocking scale remains part of transition-basin
scoring even if an experimental alternative node classifier is selected.

## Optional analysis methods

These remain available for comparison but are not defaults.

### Friction-range policy

`--friction-policy range` samples the inferred coefficient from `mu=0` to the
configured static limit. It is useful for diagnosing poses whose classification
depends on the assumed sliding friction, but it can reject valid nominal
zero-friction equilibria and is much more expensive on large catalogs.

### CWSA

`--pose-ranking csa` and/or `--robustness-method csa` enable the experimental
contact-wrench solid-angle (CWSA) analysis. By default CWSA samples 72 equal-
area gravity directions in a 5-degree spherical cap and solves the separate
floor/wall contact-wrench equilibrium at each direction. Failed directions
contribute zero; the score averages the worst sampled contact-load margin over
the cap. Its `0..1` score is absolute and dimensionless, but is not a
likelihood.

The provisional experimental CWSA robustness cutoff is `0.65`; face-face poses
still require the `0.10 g` braking safeguard. This cutoff needs calibration
against measured trials. CWSA is not applicable to some continuously reseating
rolling-contact modes because a fixed-contact sweep cannot model contact
switching; those modes retain the documented rocking fallback.

### Disturbance capacities

The `disturbance` command reports the smallest additional downhill-opposing
braking force per unit mass that first unloads a boundary contact, plus the
smallest positive or negative upset moment about each mass-principal axis. Its
upset-torque capacity is normalized by gravity and part length. Each solve
redistributes non-negative floor and wall pressure while retaining force and
moment equilibrium. First unloading need not mean complete overturning, so
these are diagnostics and supply the face-face safety gate; the default roadmap
ordering and primary classifier remain the finite rocking barrier.

## Outputs

The `roadmap` command exports:

- JSON for GUI/runtime consumption,
- editable YAML handover data,
- GraphML network data,
- PNG and SVG roadmap plots,
- per-pose thumbnails.

The runtime JSON also embeds each representative pose image as
`thumbnail_png_base64`. The `geometry_status=verified` label means the input
geometry/CAD status has been approved; it does **not** claim that every
transition has been experimentally verified.

Technical pose sheets mark floor contacts and true contact edges in green,
wall contacts and edges in orange, and common floor-wall seam contacts in red.
The roadmap YAML is intended for experimental editing; JSON is the runtime
format and GraphML supports independent network inspection.

Roadmap plots use red for X, green for Y, and blue for Z rotations. Robust
transitions are bright and solid; paths touching metastable poses are faint,
dashed, and intentionally unlabeled. Robust transition angles retain their
signed degree labels.

Generated result directories are ignored by Git. Keep experimentally approved
roadmaps in the appropriate external deployment/configuration repository when
they are ready for production use.

The `route` command finds the highest-scoring open-loop path within its action
limit (four actuations by default). It does not observe intermediate states or
replan between air impulses.

## Model limits

The analysis is quasi-static. It does not simulate drops, impacts, elastic
deformation, airflow, time-dependent friction, or collisions between multiple
workpieces. Passive tipping and capture basins use discretized seated-energy
paths. Therefore a computed pose or edge is a mechanically screened candidate,
not an experimental guarantee; production roadmaps still require trials.

## Repository layout

```text
src/chute_pose/          supported geometry, stability, plotting and roadmap code
tests/                   modern regression tests
Werkstücke_STL_grob/     active STL inputs and STEP verification models
chute-pose.ps1           shared-environment PowerShell launcher
legacy/                  archived OBJ/CSV pipeline, converters and old pose plots
```

See [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) for integration and deployment
details. This README is the authoritative guide to the supported pose pipeline.

## Development checks

```powershell
.\chute-pose.ps1 --help
python -m pytest
python -m ruff check src tests
```

## License

MIT. See [`LICENSE`](LICENSE).
