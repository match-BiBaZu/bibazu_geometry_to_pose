"""Command-line entry point for the deterministic chute-pose pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .frame import ChuteFrame
from .geometry import GeometryValidationError, inspect_mesh
from .contacts import build_pose_catalog
from .visualization import render_pose_sheets


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
    print("Angle-dependent stability filtering: pending (Step 3)")
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
    except (GeometryValidationError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
