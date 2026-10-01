# CSA / CRSA: reference calculations and experimental chute adaptation

## Sources and naming

* Ngoi, Lee & Lim (1995), *Analysing the probabilities of the natural resting
  aspects of a component with a displaced centre of gravity*, section 3,
  DOI **10.1080/00207549508904822** (user-supplied `CSAPaper.pdf`).
* Ngoi, Lye & Chen (1996), *Analysing the natural resting aspect of a prism on a
  hard surface for automated assembly*, sections 2.3–3.2, equations 4–8,
  DOI **10.1007/BF01178966** (user-supplied `CRSAPaper.pdf`).

`classical.planar_reference` implements the horizontal-plane raw weights:

```
CSA_i  = Omega_i / h_i
CRSA_i = (Omega_i - mean_j(Omega_critical_ij)) / h_i
```

For each boundary edge, the critical apex lies on the original normal through
the COM projection, at height equal to the COM-to-edge-line distance. It is
not simply the COM in the tipped geometry. A triangle-fan solid-angle formula
replaces the papers' CAD sphere/pyramid intersection calculation. It computes
the same steradians for a convex planar aspect. Holes do not automatically
become support holes: the aspect is the support envelope, using the actual
solid's COM, consistent with the displaced-COM example.

`normalize_weights` sums explicitly supplied physical multiplicities and divides
by the total. Mesh triangle counts and duplicated catalogue orientations are
not physical multiplicities. Reference tests cover analytical square/cube
solid angles, critical apex geometry, displacement, scale, rotation/translation
invariance and equivalent face counts. These are mathematical checks, **not a
reproduction of the papers' measured drop-test curves**.

## Chute extension: `reseated-patch-normal-load-v1`

This is a documented experimental extension, not a formula published in either
paper and not an experimentally calibrated landing probability. CWSA remains a
separate contact-wrench model. For backward compatibility CLI `csa` still means
CWSA; `standard_csa` selects the new CSA-based chute score, and `crsa` the new
CRSA-based chute score. GUI labels explicitly say “chute”.

Use the current `ChuteFrame`: `R_world_from_chute = Ry(beta) @ Rx(alpha)`, floor
`z=0`, wall `y=0`, body in `y,z>=0`; plane intersection is X. Gravity is transformed
once to the chute frame. The fixed plot camera is not a physical rotation.
Default GUI angles are X=45°, Y=0°. Material is assumed uniform density, as in
the existing mesh pipeline; no material-density map is inferred from an STL.

For each loaded contact plane s:

1. Form its own convex contact polygon; do not project or merge the two planes.
2. Calculate solid angle Omega_s at the COM and normal distance h_s.
3. Set c_s = max(0, up dot n_s), f_s = c_s / sum(c_s), where up = -g/|g|.
4. Define gravity-ray height H_s = h_s/c_s and CSA score
   `sum_s f_s * Omega_s / H_s`. The fractions are total normal-load proportions,
   not empirically fitted coefficients. This choice is an extension assumption.
5. For CRSA, follow outward rotations about each patch boundary-edge direction.
   Rotate the mesh about its COM and translate it back into contact with both
   planes, using `-min(y)` and `-min(z)` at every angle, as in `rocking.py`.
   This permits frictionless re-seating/contact changes rather than rotating
   through the other plane. The longitudinal COM coordinate stays fixed.
6. Scan to the first gravitational height peak, not just the 5° rocking
   excursion. A 0.5° grid over one revolution brackets the first descending
   interval; a bounded scalar refinement locates the peak. This is not proof
   of a global minimum escape saddle; narrow sub-grid extrema can be missed.
7. Convert the height rise dH to a virtual apex above the original patch:
   `COM + n_s*dH/c_s`. Compute its solid angle Omega'_sj. Average the
   nonnegative deficits over that patch's edges and combine as
   `sum_s f_s * mean_j(Omega_s - Omega'_sj) / H_s`.

For a single horizontal plane and a fixed-pivot escape before any new contact,
the saddle construction reduces to the paper's radius-height construction.
The **two-plane weights and virtual-apex mapping are modelling assumptions**.
They require physical calibration, especially when one plane takes a small
normal load or contact changes early in the motion.

Loaded line/point/degenerate patches and continuously rolling bodies are N/A,
not fabricated finite-area polygons. Undefined saddles also produce N/A. There
is no automatic rocking fallback for CSA/CRSA classification. Reasons and
per-edge lifts/critical angles are retained in data exports. For nonzero beta,
the model does not supply the force needed to hold the part along X; scores
describe tipping geometry under the prescribed setup, not full rest equilibrium.

## Ranking, normalization and classification

* Raw scores have units **sr/mm**. A single cutoff is not scale independent.
* Practical-class score = minimum across equivalent catalogue representations,
  matching the roadmap's conservative treatment of other metrics.
* Optional normalized shares divide class scores by the sum over applicable
  classes **before robust filtering**. These shares are not validated paper
  probabilities for the chute. Unsupported classes are excluded and marked N/A.
* Rocking remains the default classifier (GUI preset 0.20 mm, no braking).
* CSA/CRSA classification requires an explicit positive user-calibrated raw
  threshold. No universal threshold is claimed. N/A cannot qualify as robust.
* New ordering choices preserve rocking-based unique pose IDs. Ranks are dense,
  zero-based and tied at displayed precision (3 decimals for rocking/CWSA,
  6 for CSA/CRSA). Ranking never overwrites pose identity.

The legacy `csa` ordering option keeps its historical numbering for existing
callers. New applications should use `cwsa` explicitly.

## GUI and batch operation

Double-click `PoseRoadmapGUI.cmd`, or run `Start-PoseRoadmapGUI.ps1`. The launcher
uses this repository's local `.venv` (`uv sync --extra gui`, or create a local
venv and install `pip install -e ".[gui]"`). This repository's `WindowsLaunchers`
folder installs **BiBaZu Pose Roadmap Generator**, including its custom icon;
installation is optional and does not require BiBaZu_Big_Boi.

The GUI runs a separate Python process, so calculation/plotting does not block
the event loop. Folder selection supports individual STL files, search, select
all/none, Ctrl-click multi-selection in the workpiece table, and search filtering.
Data and plot formats are independent.
Both ordering and classification select one method; display uses independent
checkboxes. Settings and selection persist through QSettings.

For command-line batch use, `python -m chute_pose.generate --config run.json`
accepts `{"meshes": ["absolute/part.STL"], "settings": {...}}`; settings follow
`GenerationConfig` in `generate.py`. YAML-only really writes only YAML. Plots
can independently be SVG or PNG, including pose sheets. Classic CLI examples:

```
chute-pose roadmap part.STL --output-dir out --alpha 45 --beta 0 --minimum-rocking-barrier-mm 0.20 --minimum-braking-g 0 --classical-methods standard_csa crsa --comparison-plots
```

Robust-only filtering applies to data and pictures and excludes paths through
non-robust intermediate catalogue poses. Scores/ranks are computed first. The
GUI's “Include metastable poses in pose sheets” option independently adds
metastable pose cells while leaving roadmap/data robust-only. Robust-only and
combined pose sheets are saved in separate folders. Metric comparison SVG/PNG
outputs are pose sheets annotated with each selected method's value and rank;
Roadmap SVG/PNG outputs remain the full transition graph.
The current methods, units, source DOIs and run settings are recorded in data
exports; GraphML stores nested metric information as JSON strings.

Generation stages all selected files for a workpiece before publication and
atomically replaces individual files. Existing-file policy is per workpiece:
skip a populated output directory, or overwrite selected output names after
confirmation. Unselected existing formats are not deleted. Cancellation kills
the worker, keeps completed outputs, and can leave an uncommitted hidden
`.pose-stage-*` directory. No hardware connection is made.
