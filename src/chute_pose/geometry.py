"""Validated mesh loading and geometry reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import trimesh


Vector3 = NDArray[np.float64]


class GeometryValidationError(ValueError):
    """Raised when a mesh cannot support reliable mass-property analysis."""


def _vector(values: NDArray[np.floating[Any]]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return (float(array[0]), float(array[1]), float(array[2]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_oriented_normals(
    normals: NDArray[np.floating[Any]], angular_tolerance_deg: float
) -> list[Vector3]:
    cos_tolerance = float(np.cos(np.deg2rad(angular_tolerance_deg)))
    result: list[Vector3] = []

    # Copy because recent trimesh versions expose cached arrays as read-only.
    for raw_normal in np.asarray(normals, dtype=float).copy():
        length = float(np.linalg.norm(raw_normal))
        if length == 0.0:
            continue
        normal = raw_normal / length
        if not any(float(np.dot(normal, known)) >= cos_tolerance for known in result):
            result.append(normal)
    return result


@dataclass(frozen=True, slots=True)
class GeometryReport:
    source: str
    sha256: str
    units: str
    vertex_count: int
    face_count: int
    watertight: bool
    winding_consistent: bool
    volume_mm3: float
    surface_area_mm2: float
    bounds_min_mm: tuple[float, float, float]
    bounds_max_mm: tuple[float, float, float]
    extents_mm: tuple[float, float, float]
    center_mass_mm: tuple[float, float, float]
    surface_centroid_mm: tuple[float, float, float]
    hull_vertex_count: int
    hull_face_count: int
    hull_volume_mm3: float
    hull_plane_count: int
    uniform_density_assumed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_solid_mesh(
    mesh_path: str | Path,
    *,
    units: str = "mm",
) -> trimesh.Trimesh:
    """Load and strictly validate a solid mesh without silently repairing it.

    Step 1 deliberately rejects open or inconsistently wound meshes. Silent
    repair would make the center of mass dependent on library heuristics.
    """

    path = Path(mesh_path).expanduser().resolve()
    if not path.is_file():
        raise GeometryValidationError(f"Mesh file does not exist: {path}")
    if units != "mm":
        raise GeometryValidationError("Step 1 currently accepts millimetres only.")

    loaded = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise GeometryValidationError(f"Expected one triangle mesh in {path}.")
    mesh = loaded

    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise GeometryValidationError("The mesh does not contain a closed 3-D solid.")
    if not mesh.is_watertight:
        raise GeometryValidationError(
            "The mesh is not watertight; its volume and center of mass are unreliable."
        )
    if not mesh.is_winding_consistent:
        raise GeometryValidationError(
            "The mesh winding is inconsistent; outward surface normals are ambiguous."
        )
    if not mesh.is_volume or float(mesh.volume) <= 0.0:
        raise GeometryValidationError("The mesh does not describe a positive closed volume.")

    return mesh


def inspect_mesh(
    mesh_path: str | Path,
    *,
    units: str = "mm",
    angular_tolerance_deg: float = 0.1,
) -> GeometryReport:
    """Load a solid mesh and return its reproducible geometry report."""

    path = Path(mesh_path).expanduser().resolve()
    if not math_is_positive(angular_tolerance_deg):
        raise GeometryValidationError("Angular tolerance must be positive.")

    mesh = load_solid_mesh(path, units=units)

    hull = mesh.convex_hull
    hull_normals = _unique_oriented_normals(
        hull.face_normals, angular_tolerance_deg=angular_tolerance_deg
    )

    return GeometryReport(
        source=str(path),
        sha256=_sha256(path),
        units=units,
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        volume_mm3=float(mesh.volume),
        surface_area_mm2=float(mesh.area),
        bounds_min_mm=_vector(mesh.bounds[0]),
        bounds_max_mm=_vector(mesh.bounds[1]),
        extents_mm=_vector(mesh.extents),
        center_mass_mm=_vector(mesh.center_mass),
        surface_centroid_mm=_vector(mesh.centroid),
        hull_vertex_count=int(len(hull.vertices)),
        hull_face_count=int(len(hull.faces)),
        hull_volume_mm3=float(hull.volume),
        hull_plane_count=len(hull_normals),
    )


def math_is_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)
