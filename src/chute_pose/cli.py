"""Command-line entry point for the deterministic chute-pose pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .frame import ChuteFrame
from .geometry import GeometryValidationError, inspect_mesh, load_solid_mesh
from .contacts import build_pose_catalog
from .stability import analyze_pose_stability
from .disturbance import (
    analyze_disturbance_robustness,
    filter_disturbance_robustness,
)
from .equivalence import cluster_practical_contact_poses
from .rocking import (
    analyze_rocking_barriers,
    filter_finite_disturbance_robustness,
)
from .symmetry import detect_rotational_symmetry, reduce_catalog_by_symmetry
from .step_verification import StepSupportUnavailable, verify_step_symmetry
from .visualization import render_pose_sheets
from .roadmap import (
    build_pose_roadmap,
    export_pose_roadmap,
    find_best_route,
    load_roadmap_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chute-pose")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate one part mesh and print the Step-1 report."
    )
    inspect_parser.add_argument("mesh", type=Path)
    inspect_parser.add_argument("--alpha", type=float, default=45.0)
    inspect_parser.add_argument("--beta", type=float, default=20.0)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    catalog_parser = subparsers.add_parser(
        "catalog", help="Enumerate theoretical simultaneous floor-wall poses."
    )
    catalog_parser.add_argument("mesh", type=Path)
    catalog_parser.add_argument("--json", action="store_true", dest="as_json")

    render_parser = subparsers.add_parser(
        "render", help="Render theoretical poses as grouped PNG contact sheets."
    )
    render_parser.add_argument("mesh", type=Path)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--poses-per-sheet", type=int, default=24)
    render_parser.add_argument("--columns", type=int, default=6)
    render_parser.add_argument("--dpi", type=int, default=180)

    stability_parser = subparsers.add_parser(
        "stability", help="Filter theoretical poses by sliding force/moment stability."
    )
    stability_parser.add_argument("mesh", type=Path)
    stability_parser.add_argument("--alpha", type=float, default=45.0)
    stability_parser.add_argument("--beta", type=float, default=20.0)
    stability_parser.add_argument("--onset-alpha", type=float, default=45.0)
    stability_parser.add_argument("--onset-beta", type=float, default=15.0)
    stability_parser.add_argument("--mu-samples", type=int, default=11)
    stability_parser.add_argument("--symmetry-tolerance-mm", type=float)
    stability_parser.add_argument("--json", action="store_true", dest="as_json")
    stability_parser.add_argument(
        "--render-output-dir",
        type=Path,
        help="Optionally render only poses stable at every sampled mu value.",
    )

    symmetry_parser = subparsers.add_parser(
        "symmetry", help="Detect STL symmetry, verify it with STEP, and group poses."
    )
    symmetry_parser.add_argument("mesh", type=Path)
    symmetry_parser.add_argument("--step", type=Path)
    symmetry_parser.add_argument("--tolerance-mm", type=float)
    symmetry_parser.add_argument("--angular-tolerance-deg", type=float, default=0.25)
    symmetry_parser.add_argument("--json", action="store_true", dest="as_json")

    disturbance_parser = subparsers.add_parser(
        "disturbance",
        help="Filter nominal poses by critical braking-force and upset-torque reserve.",
    )
    disturbance_parser.add_argument("mesh", type=Path)
    disturbance_parser.add_argument(
        "--pose-ids", help="Comma-separated pose ids; default is the nominal stable set."
    )
    disturbance_parser.add_argument("--alpha", type=float, default=45.0)
    disturbance_parser.add_argument("--beta", type=float, default=20.0)
    disturbance_parser.add_argument("--onset-alpha", type=float, default=45.0)
    disturbance_parser.add_argument("--onset-beta", type=float, default=15.0)
    disturbance_parser.add_argument("--mu-samples", type=int, default=11)
    disturbance_parser.add_argument("--minimum-braking-g", type=float, default=0.10)
    disturbance_parser.add_argument(
        "--minimum-torque-normalized", type=float, default=0.02
    )
    disturbance_parser.add_argument(
        "--rocking-excursion-deg", type=float, default=5.0
    )
    disturbance_parser.add_argument("--rocking-angle-steps", type=int, default=20)
    disturbance_parser.add_argument("--rocking-axis-samples", type=int, default=2048)
    disturbance_parser.add_argument(
        "--minimum-rocking-barrier-mm", type=float, default=0.20
    )
    disturbance_parser.add_argument(
        "--minimum-face-face-braking-g", type=float, default=0.10
    )
    disturbance_parser.add_argument("--symmetry-tolerance-mm", type=float)
    disturbance_parser.add_argument(
        "--contact-angle-tolerance-deg", type=float, default=1.0
    )
    disturbance_parser.add_argument(
        "--contact-displacement-tolerance-mm", type=float, default=0.5
    )
    disturbance_parser.add_argument("--render-output-dir", type=Path)
    disturbance_parser.add_argument("--json", action="store_true", dest="as_json")

    roadmap_parser = subparsers.add_parser(
        "roadmap",
        help="Build the robust/metastable pose roadmap and export JSON/GraphML/images.",
    )
    roadmap_parser.add_argument("mesh", type=Path)
    roadmap_parser.add_argument("--output-dir", type=Path, required=True)
    roadmap_parser.add_argument("--alpha", type=float, default=45.0)
    roadmap_parser.add_argument("--beta", type=float, default=20.0)
    roadmap_parser.add_argument("--onset-alpha", type=float, default=45.0)
    roadmap_parser.add_argument("--onset-beta", type=float, default=15.0)
    roadmap_parser.add_argument("--symmetry-tolerance-mm", type=float, default=0.5)
    roadmap_parser.add_argument("--axis-tolerance-deg", type=float, default=1.0)
    roadmap_parser.add_argument(
        "--surface-displacement-tolerance-mm", type=float, default=0.5
    )
    roadmap_parser.add_argument(
        "--minimum-rocking-barrier-mm", type=float, default=0.20
    )
    roadmap_parser.add_argument(
        "--minimum-face-face-braking-g", type=float, default=0.10
    )
    roadmap_parser.add_argument(
        "--geometry-status",
        choices=("provisional", "verified"),
        default="provisional",
    )
    roadmap_parser.add_argument("--json", action="store_true", dest="as_json")

    route_parser = subparsers.add_parser(
        "route", help="Find the highest-scoring open-loop route in a roadmap JSON."
    )
    route_parser.add_argument("roadmap", type=Path)
    route_parser.add_argument("--start-pose", type=int, required=True)
    route_parser.add_argument("--target-pose", type=int, required=True)
    route_parser.add_argument("--max-actions", type=int, default=4)
    route_parser.add_argument("--output", type=Path)
    return parser


def _inspect(args: argparse.Namespace) -> int:
    frame = ChuteFrame(alpha_deg=args.alpha, beta_deg=args.beta)
    report = inspect_mesh(args.mesh)
    gravity = frame.gravity_chute()

    result = {
        "coordinate_system": {
            "handedness": "right-handed",
            "x": "downhill along chute",
            "y": "away from wall; admissible interior y >= 0",
            "z": "away from floor; admissible interior z >= 0",
            "floor": "z = 0",
            "wall": "y = 0",
            "floor_wall_seam": "(x, 0, 0)",
            "contact_modes": ["floor_wall"],
        },
        "orientation": {
            "alpha_deg_moved_x": frame.alpha_deg,
            "beta_deg_original_y": frame.beta_deg,
            "rotation_order": "R_y(beta) @ R_x(alpha)",
            "gravity_chute_m_s2": [float(value) for value in gravity],
        },
        "geometry": report.to_dict(),
    }

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Mesh: {report.source}")
        print(f"SHA-256: {report.sha256}")
        print(f"Units: {report.units}")
        print(f"Extents: {report.extents_mm} mm")
        print(f"Volume: {report.volume_mm3:.6f} mm^3")
        print(f"Center of mass: {report.center_mass_mm} mm")
        print(
            "Convex hull: "
            f"{report.hull_vertex_count} vertices, "
            f"{report.hull_face_count} triangles, "
            f"{report.hull_plane_count} oriented planes"
        )
        print(
            "Chute angles: "
            f"alpha={frame.alpha_deg:g} deg about moved X, "
            f"beta={frame.beta_deg:g} deg about original Y"
        )
        print(
            "Gravity in chute frame: "
            f"({gravity[0]:.6f}, {gravity[1]:.6f}, {gravity[2]:.6f}) m/s^2"
        )
        print("Required contact mode: floor_wall")
    return 0


def _catalog(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    if args.as_json:
        print(json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False))
        return 0

    type_counts: dict[str, int] = {}
    for pose in catalog.poses:
        key = f"{pose.floor_contact_type}-{pose.wall_contact_type}"
        type_counts[key] = type_counts.get(key, 0) + 1

    print(f"Mesh: {catalog.source}")
    print(f"Convex support faces: {len(catalog.support_faces)}")
    print(f"Theoretical floor-wall poses: {len(catalog.poses)}")
    for contact_type, count in sorted(type_counts.items()):
        print(f"  {contact_type}: {count}")
    print("Point contacts excluded: yes")
    print("Edge-edge contacts excluded as non-isolated: yes")
    print("Run the 'stability' command for angle-dependent filtering (Step 3).")
    return 0


def _render(args: argparse.Namespace) -> int:
    sheets = render_pose_sheets(
        args.mesh,
        args.output_dir,
        poses_per_sheet=args.poses_per_sheet,
        columns=args.columns,
        dpi=args.dpi,
    )
    print(f"Rendered {len(sheets)} contact sheets:")
    for sheet in sheets:
        first_pose = sheet.pose_ids[0]
        last_pose = sheet.pose_ids[-1]
        print(f"  {sheet.path} (poses {first_pose}-{last_pose})")
    return 0


def _stability(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    analysis = analyze_pose_stability(
        args.mesh,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        mu_samples=args.mu_samples,
        catalog=catalog,
    )
    detected_symmetry = detect_rotational_symmetry(
        args.mesh, tolerance_mm=args.symmetry_tolerance_mm
    )
    verification = None
    if args.symmetry_tolerance_mm is not None:
        symmetry = detected_symmetry
        symmetry_policy = "explicit_practical_tolerance"
    elif detected_symmetry.order == 1:
        symmetry = detected_symmetry
        symmetry_policy = "no_nontrivial_symmetry"
    else:
        step_path = _matching_step_path(args.mesh)
        try:
            verification = (
                verify_step_symmetry(step_path, detected_symmetry)
                if step_path is not None
                else None
            )
        except StepSupportUnavailable:
            verification = None
        if verification is not None and verification.exact_confirmed:
            symmetry = detected_symmetry
            symmetry_policy = "exact_step_confirmation"
        else:
            symmetry = replace(
                detected_symmetry,
                symbol="C1",
                elements=(detected_symmetry.elements[0],),
            )
            symmetry_policy = "not_merged_without_exact_step_confirmation"
    reduced = reduce_catalog_by_symmetry(catalog, symmetry)
    stable_ids = set(analysis.stable_pose_ids)
    stable_class_representatives = tuple(
        min(stable_ids.intersection(value.pose_ids))
        for value in reduced.classes
        if stable_ids.intersection(value.pose_ids)
    )
    stable_pose_labels = {
        min(stable_ids.intersection(value.pose_ids)): (
            "Klasse " + "/".join(str(pose_id) for pose_id in value.pose_ids)
        )
        for value in reduced.classes
        if stable_ids.intersection(value.pose_ids)
    }
    if args.render_output_dir is not None:
        part_name = args.mesh.stem
        render_pose_sheets(
            args.mesh,
            args.render_output_dir,
            pose_ids=stable_class_representatives,
            sheet_title=(
                f"{part_name}: quasistatisch zulaessige Gleitlagen "
                f"bei alpha={args.alpha:g} deg, "
                f"beta={args.beta:g} deg"
            ),
            filename_prefix=f"{part_name}_quasistatic",
            pose_labels=stable_pose_labels,
        )

    if args.as_json:
        print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
        return 0

    type_counts: dict[str, list[int]] = {}
    for result in analysis.poses:
        key = f"{result.floor_contact_type}-{result.wall_contact_type}"
        counts = type_counts.setdefault(key, [0, 0])
        counts[1] += 1
        if result.stable_across_range:
            counts[0] += 1

    estimate = analysis.friction_estimate
    print(f"Mesh: {analysis.source}")
    print(
        "Chute angles: "
        f"alpha={analysis.alpha_deg:g} deg, beta={analysis.beta_deg:g} deg"
    )
    print(
        "Friction estimate from slide onset: "
        f"mu_s={estimate.mu_static_estimate:.6f} "
        f"(alpha={estimate.onset_alpha_deg:g} deg, beta={estimate.onset_beta_deg:g} deg)"
    )
    print(
        f"Robust interval: 0 <= mu <= {estimate.mu_static_estimate:.6f} "
        f"at {len(analysis.mu_values)} samples"
    )
    print(
        f"Quasi-statically admissible at every sampled coefficient: "
        f"{len(analysis.stable_pose_ids)}/{len(analysis.poses)}"
    )
    for contact_type, (stable_count, total_count) in sorted(type_counts.items()):
        print(f"  {contact_type}: {stable_count}/{total_count}")
    print(
        "Quasi-static pose IDs: "
        + ", ".join(str(value) for value in analysis.stable_pose_ids)
    )
    print(
        "Friction-dependent candidates: "
        f"{len(analysis.friction_dependent_pose_ids)}; "
        f"rejected at every sample: {len(analysis.rejected_pose_ids)}"
    )
    print(
        f"Applied rotation symmetry: {symmetry.symbol} "
        f"(order {symmetry.order}, tolerance {symmetry.tolerance_mm:.6g} mm)"
    )
    print(f"Symmetry merge policy: {symmetry_policy}")
    if detected_symmetry.order > symmetry.order:
        print(
            f"Unmerged STL candidate: {detected_symmetry.symbol} "
            "(requires exact STEP confirmation or explicit tolerance)"
        )
    print(
        "Quasi-static pose classes after symmetry grouping: "
        f"{len(stable_class_representatives)}"
    )
    if args.render_output_dir is not None:
        print(f"Rendered quasi-static pose classes to: {args.render_output_dir.resolve()}")
    return 0


def _matching_step_path(mesh_path: Path) -> Path | None:
    for candidate in mesh_path.parent.glob(f"{mesh_path.stem}.*"):
        if candidate.suffix.lower() in {".step", ".stp"}:
            return candidate
    return None


def _symmetry(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    symmetry = detect_rotational_symmetry(
        args.mesh, tolerance_mm=args.tolerance_mm
    )
    reduced = reduce_catalog_by_symmetry(
        catalog,
        symmetry,
        angular_tolerance_deg=args.angular_tolerance_deg,
    )
    step_path = args.step or _matching_step_path(args.mesh)
    verification = verify_step_symmetry(step_path, symmetry) if step_path else None

    if args.as_json:
        result = reduced.to_dict()
        result["step_verification"] = (
            verification.to_dict() if verification is not None else None
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    nontrivial_classes = [value for value in reduced.classes if len(value.pose_ids) > 1]
    print(f"Mesh: {catalog.source}")
    print(
        f"STL symmetry candidate: {symmetry.symbol}, order {symmetry.order}, "
        f"practical tolerance {symmetry.tolerance_mm:.6g} mm"
    )
    if verification is None:
        print("STEP verification: no matching STEP file found")
    else:
        print(f"STEP verification: {verification.status}")
        if verification.checks:
            maximum_difference = max(
                check.relative_symmetric_difference for check in verification.checks
            )
            print(
                "Maximum STEP symmetric-volume difference: "
                f"{100.0 * maximum_difference:.6f}%"
            )
    print(
        f"Pose rotations: {len(catalog.poses)} -> "
        f"{len(reduced.classes)} physical pose classes"
    )
    print(f"Non-singleton equivalence classes: {len(nontrivial_classes)}")
    for value in nontrivial_classes:
        print(
            f"  class {value.class_id}, representative {value.representative_pose_id}: "
            + ", ".join(str(pose_id) for pose_id in value.pose_ids)
        )
    return 0


def _parse_pose_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        pose_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--pose-ids must be a comma-separated list of integers.") from exc
    if not pose_ids:
        raise ValueError("--pose-ids must contain at least one integer.")
    return pose_ids


def _disturbance(args: argparse.Namespace) -> int:
    catalog = build_pose_catalog(args.mesh)
    requested_pose_ids = _parse_pose_ids(args.pose_ids)
    if requested_pose_ids is None:
        nominal = analyze_pose_stability(
            args.mesh,
            alpha_deg=args.alpha,
            beta_deg=args.beta,
            onset_alpha_deg=args.onset_alpha,
            onset_beta_deg=args.onset_beta,
            mu_samples=args.mu_samples,
            catalog=catalog,
        )
        requested_pose_ids = nominal.stable_pose_ids
    analysis = analyze_disturbance_robustness(
        args.mesh,
        pose_ids=requested_pose_ids,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        mu_samples=args.mu_samples,
        catalog=catalog,
    )
    filtered = filter_disturbance_robustness(
        analysis,
        minimum_braking_g=args.minimum_braking_g,
        minimum_torque_normalized=args.minimum_torque_normalized,
    )
    rocking = analyze_rocking_barriers(
        args.mesh,
        pose_ids=requested_pose_ids,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        excursion_deg=args.rocking_excursion_deg,
        angle_steps=args.rocking_angle_steps,
        axis_samples=args.rocking_axis_samples,
        catalog=catalog,
    )
    finite_filtered = filter_finite_disturbance_robustness(
        rocking,
        analysis,
        catalog,
        minimum_barrier_height_mm=args.minimum_rocking_barrier_mm,
        minimum_face_face_braking_g=args.minimum_face_face_braking_g,
    )
    mesh = load_solid_mesh(args.mesh)
    vertices_centered = np.asarray(mesh.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    symmetry = (
        detect_rotational_symmetry(
            args.mesh, tolerance_mm=args.symmetry_tolerance_mm
        )
        if args.symmetry_tolerance_mm is not None
        else None
    )
    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered,
        requested_pose_ids,
        symmetry=symmetry,
        angular_tolerance_deg=args.contact_angle_tolerance_deg,
        surface_displacement_tolerance_mm=max(
            args.contact_displacement_tolerance_mm,
            symmetry.tolerance_mm if symmetry is not None else 0.0,
        ),
    )
    capacities = {value.pose_id: value for value in analysis.capacities}
    barriers = {value.pose_id: value for value in rocking.barriers}
    accepted_ids = set(finite_filtered.accepted_pose_ids)
    robust_classes = tuple(
        pose_class
        for pose_class in clustering.classes
        if all(pose_id in accepted_ids for pose_id in pose_class.pose_ids)
    )
    robust_representation_count = sum(
        len(pose_class.pose_ids) for pose_class in robust_classes
    )

    if args.render_output_dir is not None:
        labels = {
            pose_class.representative_pose_id: (
                "Klasse " + "/".join(str(value) for value in pose_class.pose_ids)
            )
            for pose_class in robust_classes
        }
        render_pose_sheets(
            args.mesh,
            args.render_output_dir,
            pose_ids=labels,
            sheet_title=(
                f"{args.mesh.stem}: stoerfeste Gleitposen "
                f"bei alpha={args.alpha:g} deg, beta={args.beta:g} deg"
            ),
            filename_prefix=f"{args.mesh.stem}_disturbance_robust",
            pose_labels=labels,
        )

    if args.as_json:
        result = analysis.to_dict()
        result["filter"] = filtered.to_dict()
        result["rocking"] = rocking.to_dict()
        result["finite_disturbance_filter"] = finite_filtered.to_dict()
        result["practical_clustering"] = clustering.to_dict()
        result["robust_practical_classes"] = [
            pose_class.to_dict() for pose_class in robust_classes
        ]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Mesh: {analysis.source}")
    print(f"Nominal input poses: {len(requested_pose_ids)}")
    print(
        "Finite disturbance thresholds: "
        f"rocking barrier >= {finite_filtered.minimum_barrier_height_mm:.6g} mm; "
        "for face-face additionally braking >= "
        f"{finite_filtered.minimum_face_face_braking_g:.6g} g"
    )
    print(
        f"Disturbance-robust representations in complete classes: "
        f"{robust_representation_count}; practical pose classes: "
        f"{len(robust_classes)}"
    )
    for pose_class in robust_classes:
        member_capacities = [capacities[value] for value in pose_class.pose_ids]
        print(
            "  "
            + "/".join(str(value) for value in pose_class.pose_ids)
            + f": braking={min(value.critical_braking_g for value in member_capacities):.6f} g"
            + ", torque="
            + f"{min(value.critical_torque_normalized for value in member_capacities):.6f}"
            + ", rocking="
            + f"{min(barriers[value].barrier_height_mm for value in pose_class.pose_ids):.6f} mm"
        )
    if args.render_output_dir is not None:
        print(f"Rendered robust classes to: {args.render_output_dir.resolve()}")
    return 0


def _roadmap(args: argparse.Namespace) -> int:
    result = build_pose_roadmap(
        args.mesh,
        alpha_deg=args.alpha,
        beta_deg=args.beta,
        onset_alpha_deg=args.onset_alpha,
        onset_beta_deg=args.onset_beta,
        symmetry_tolerance_mm=args.symmetry_tolerance_mm,
        angular_tolerance_deg=args.axis_tolerance_deg,
        surface_displacement_tolerance_mm=(
            args.surface_displacement_tolerance_mm
        ),
        robust_barrier_threshold_mm=args.minimum_rocking_barrier_mm,
        minimum_face_face_braking_g=args.minimum_face_face_braking_g,
        geometry_status=args.geometry_status,
    )
    paths = export_pose_roadmap(result, args.output_dir)
    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    robust_count = sum(node.kind == "robust" for node in result.nodes)
    metastable_count = sum(node.kind == "metastable" for node in result.nodes)
    actuated_count = sum(edge.transition_kind == "actuated" for edge in result.edges)
    passive_count = sum(edge.transition_kind == "passive_tip" for edge in result.edges)
    print(f"Mesh: {result.source}")
    print(
        f"Roadmap nodes: {len(result.nodes)} "
        f"({robust_count} robust, {metastable_count} metastable)"
    )
    print(
        f"Directed transitions: {actuated_count} actuated, "
        f"{passive_count} passive"
    )
    if result.unresolved_metastable_node_ids:
        print(
            "Unresolved metastable nodes: "
            + ", ".join(str(value) for value in result.unresolved_metastable_node_ids)
        )
    print(f"Geometry status: {result.geometry_status}")
    for path in paths:
        print(f"  {path}")
    return 0


def _route(args: argparse.Namespace) -> int:
    roadmap = load_roadmap_json(args.roadmap)
    route = find_best_route(
        roadmap,
        args.start_pose,
        args.target_pose,
        max_actions=args.max_actions,
    )
    payload = route.to_dict()
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
        print(destination)
    else:
        print(output, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "catalog":
            return _catalog(args)
        if args.command == "render":
            return _render(args)
        if args.command == "stability":
            return _stability(args)
        if args.command == "symmetry":
            return _symmetry(args)
        if args.command == "disturbance":
            return _disturbance(args)
        if args.command == "roadmap":
            return _roadmap(args)
        if args.command == "route":
            return _route(args)
    except (GeometryValidationError, StepSupportUnavailable, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
