"""Command-line entry point for the deterministic chute-pose pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .frame import ChuteFrame
from .geometry import GeometryValidationError, inspect_mesh


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
    except (GeometryValidationError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

