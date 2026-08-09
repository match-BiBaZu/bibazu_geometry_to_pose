"""Exact theoretical floor-wall contact-pose enumeration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .geometry import load_solid_mesh


Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SupportFace:
    """One maximal coplanar polygon on the oriented convex hull."""

    face_id: int
    normal_part: tuple[float, float, float]
    plane_offset_mm: float
    vertex_indices: tuple[int, ...]
    area_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContactPose:
    """A theoretical orientation with non-point floor and wall contact."""

    pose_id: int
    quaternion_xyzw: tuple[float, float, float, float]
    rotation_chute_from_part: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation_to_corner_mm: tuple[float, float, float]
    floor_contact_dimension: int
    wall_contact_dimension: int
    floor_contact_vertex_indices: tuple[int, ...]
    wall_contact_vertex_indices: tuple[int, ...]
    floor_mesh_contact_vertex_indices: tuple[int, ...]
    wall_mesh_contact_vertex_indices: tuple[int, ...]
    floor_mesh_contact_edges: tuple[tuple[int, int], ...]
    wall_mesh_contact_edges: tuple[tuple[int, int], ...]
    floor_contact_topology: str
    wall_contact_topology: str
    floor_face_ids: tuple[int, ...]
    wall_face_ids: tuple[int, ...]
    provenance: tuple[str, ...]

    @property
    def floor_contact_type(self) -> str:
        return _dimension_name(self.floor_contact_dimension)

    @property
    def wall_contact_type(self) -> str:
        return _dimension_name(self.wall_contact_dimension)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["floor_contact_type"] = self.floor_contact_type
        result["wall_contact_type"] = self.wall_contact_type
        return result


@dataclass(frozen=True, slots=True)
class PoseCatalog:
    """All isolated, theoretical face-edge/face contact orientations."""

    source: str
    units: str
    center_mass_part_mm: tuple[float, float, float]
    support_faces: tuple[SupportFace, ...]
    poses: tuple[ContactPose, ...]
    contact_tolerance_mm: float
    rotation_tolerance_rad: float
    point_contacts_excluded: bool = True
    edge_edge_contacts_excluded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "units": self.units,
            "center_mass_part_mm": self.center_mass_part_mm,
            "support_faces": [face.to_dict() for face in self.support_faces],
            "poses": [pose.to_dict() for pose in self.poses],
            "contact_tolerance_mm": self.contact_tolerance_mm,
            "rotation_tolerance_rad": self.rotation_tolerance_rad,
            "point_contacts_excluded": self.point_contacts_excluded,
            "edge_edge_contacts_excluded": self.edge_edge_contacts_excluded,
        }


def _dimension_name(dimension: int) -> str:
    return {0: "point", 1: "edge", 2: "face"}.get(dimension, "invalid")


def _rotation_axis(axis: int, angle: float) -> Matrix3:
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == 1:  # Y
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == 2:  # Z
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported rotation axis: {axis}")


def _rotation_from_to(source: Vector3, target: Vector3) -> Matrix3:
    a = np.asarray(source, dtype=float).copy()
    b = np.asarray(target, dtype=float).copy()
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(math.pi * axis).as_matrix()

    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(math.acos(dot) * axis).as_matrix()


def _extract_support_faces(
    hull,
    *,
    angular_tolerance_deg: float,
    distance_tolerance_mm: float,
) -> tuple[SupportFace, ...]:
    normals = np.asarray(hull.face_normals, dtype=float).copy()
    triangles = np.asarray(hull.triangles, dtype=float).copy()
    faces = np.asarray(hull.faces, dtype=int).copy()
    areas = np.asarray(hull.area_faces, dtype=float).copy()
    cos_tolerance = float(np.cos(np.deg2rad(angular_tolerance_deg)))

    groups: list[dict[str, Any]] = []
    for triangle_index, (normal, triangle) in enumerate(zip(normals, triangles)):
        normal /= np.linalg.norm(normal)
        offset = float(np.dot(normal, triangle[0]))
        match = None
        for group in groups:
            if (
                float(np.dot(normal, group["normal"])) >= cos_tolerance
                and abs(offset - group["offset"]) <= distance_tolerance_mm
            ):
                match = group
                break
        if match is None:
            match = {
                "normal": normal,
                "offset": offset,
                "vertices": set(),
                "area": 0.0,
            }
            groups.append(match)
        match["vertices"].update(int(index) for index in faces[triangle_index])
        match["area"] += float(areas[triangle_index])

    groups.sort(
        key=lambda group: (
            *np.round(group["normal"], 10).tolist(),
            round(group["offset"], 10),
        )
    )
    return tuple(
        SupportFace(
            face_id=face_id,
            normal_part=tuple(float(value) for value in group["normal"]),
            plane_offset_mm=float(group["offset"]),
            vertex_indices=tuple(sorted(group["vertices"])),
            area_mm2=float(group["area"]),
        )
        for face_id, group in enumerate(groups)
    )


def _unique_points(points: NDArray[np.float64], tolerance: float) -> NDArray[np.float64]:
    scale = max(tolerance, np.finfo(float).eps)
    keys = np.round(points / scale).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def _silhouette_edges(
    rotated_vertices: NDArray[np.float64], projection_axes: tuple[int, int], tolerance: float
) -> Iterable[tuple[Vector3, Vector3]]:
    projected = _unique_points(rotated_vertices[:, projection_axes], tolerance)
    if len(projected) < 3:
        return ()
    hull_2d = ConvexHull(projected)
    polygon = projected[hull_2d.vertices]
    return tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def _affine_dimension(points: NDArray[np.float64], tolerance: float) -> int:
    if len(points) == 0:
        return -1
    if len(points) == 1:
        return 0
    centered = points - points[0]
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return int(np.count_nonzero(singular_values > tolerance))


def _contact_topology(
    contact_indices: NDArray[np.int64],
    mesh_edges: NDArray[np.int64],
    mesh_faces: NDArray[np.int64],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Describe connected mesh features in one geometric support set.

    Convex support dimension alone cannot distinguish a real face from, for
    example, an edge plus a remote outlet point.  This routine uses the full
    STL adjacency and deliberately reports disconnected point supports.
    """

    contact_set = {int(index) for index in contact_indices}
    if not contact_set:
        return "none", ()
    contact_edges = tuple(
        sorted(
            tuple(sorted((int(first), int(second))))
            for first, second in mesh_edges
            if int(first) in contact_set and int(second) in contact_set
        )
    )
    adjacency = {index: set() for index in contact_set}
    for first, second in contact_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    components: list[set[int]] = []
    remaining = set(contact_set)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current].intersection(remaining)
            remaining.difference_update(neighbors)
            component.update(neighbors)
            stack.extend(neighbors)
        components.append(component)

    pieces: list[str] = []
    for component in components:
        contains_face = any(
            all(int(vertex) in component for vertex in face) for face in mesh_faces
        )
        contains_edge = any(
            first in component and second in component
            for first, second in contact_edges
        )
        if contains_face:
            pieces.append("face")
        elif contains_edge:
            pieces.append("edge")
        else:
            pieces.extend("point" for _ in component)

    counts = {piece: pieces.count(piece) for piece in ("face", "edge", "point")}
    terms: list[str] = []
    for piece in ("face", "edge", "point"):
        count = counts[piece]
        if count == 1:
            terms.append(piece)
        elif count > 1:
            terms.append(f"{count}-{piece}")
    return "+".join(terms), contact_edges


def _canonical_quaternion(rotation: Matrix3) -> tuple[float, float, float, float]:
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < -1e-14:
        quaternion = -quaternion
    elif abs(quaternion[3]) <= 1e-14:
        first_nonzero = next((value for value in quaternion[:3] if abs(value) > 1e-14), 1.0)
        if first_nonzero < 0.0:
            quaternion = -quaternion
    return tuple(float(value) for value in quaternion)


def _face_ids_in_contact(
    contact_indices: NDArray[np.int64], support_faces: tuple[SupportFace, ...]
) -> tuple[int, ...]:
    contact_set = set(int(index) for index in contact_indices)
    return tuple(
        face.face_id
        for face in support_faces
        if set(face.vertex_indices).issubset(contact_set)
    )


def _make_candidate(
    rotation: Matrix3,
    vertices_centered: NDArray[np.float64],
    mesh_vertices_centered: NDArray[np.float64],
    mesh_edges: NDArray[np.int64],
    mesh_faces: NDArray[np.int64],
    support_faces: tuple[SupportFace, ...],
    contact_tolerance_mm: float,
    provenance: str,
) -> ContactPose | None:
    rotated = (rotation @ vertices_centered.T).T
    min_y = float(np.min(rotated[:, 1]))
    min_z = float(np.min(rotated[:, 2]))
    floor_indices = np.flatnonzero(rotated[:, 2] <= min_z + contact_tolerance_mm)
    wall_indices = np.flatnonzero(rotated[:, 1] <= min_y + contact_tolerance_mm)

    floor_dimension = _affine_dimension(
        rotated[floor_indices][:, (0, 1)], contact_tolerance_mm
    )
    wall_dimension = _affine_dimension(
        rotated[wall_indices][:, (0, 2)], contact_tolerance_mm
    )

    # Pure point contacts are transitions. Edge-edge configurations have a
    # remaining rotational degree of freedom and are not isolated poses.
    if floor_dimension < 1 or wall_dimension < 1:
        return None
    if floor_dimension < 2 and wall_dimension < 2:
        return None

    rotated_mesh = (rotation @ mesh_vertices_centered.T).T
    mesh_floor_indices = np.flatnonzero(
        rotated_mesh[:, 2]
        <= float(np.min(rotated_mesh[:, 2])) + contact_tolerance_mm
    )
    mesh_wall_indices = np.flatnonzero(
        rotated_mesh[:, 1]
        <= float(np.min(rotated_mesh[:, 1])) + contact_tolerance_mm
    )
    floor_topology, floor_mesh_edges = _contact_topology(
        mesh_floor_indices, mesh_edges, mesh_faces
    )
    wall_topology, wall_mesh_edges = _contact_topology(
        mesh_wall_indices, mesh_edges, mesh_faces
    )

    matrix_tuple = tuple(
        tuple(float(value) for value in row) for row in np.asarray(rotation)
    )
    return ContactPose(
        pose_id=-1,
        quaternion_xyzw=_canonical_quaternion(rotation),
        rotation_chute_from_part=matrix_tuple,  # type: ignore[arg-type]
        translation_to_corner_mm=(0.0, -min_y, -min_z),
        floor_contact_dimension=floor_dimension,
        wall_contact_dimension=wall_dimension,
        floor_contact_vertex_indices=tuple(int(index) for index in floor_indices),
        wall_contact_vertex_indices=tuple(int(index) for index in wall_indices),
        floor_mesh_contact_vertex_indices=tuple(
            int(index) for index in mesh_floor_indices
        ),
        wall_mesh_contact_vertex_indices=tuple(
            int(index) for index in mesh_wall_indices
        ),
        floor_mesh_contact_edges=floor_mesh_edges,
        wall_mesh_contact_edges=wall_mesh_edges,
        floor_contact_topology=floor_topology,
        wall_contact_topology=wall_topology,
        floor_face_ids=_face_ids_in_contact(floor_indices, support_faces),
        wall_face_ids=_face_ids_in_contact(wall_indices, support_faces),
        provenance=(provenance,),
    )


def _rotation_distance(a: ContactPose, b: ContactPose) -> float:
    dot = abs(float(np.dot(a.quaternion_xyzw, b.quaternion_xyzw)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _add_candidate(
    candidates: list[ContactPose], candidate: ContactPose | None, tolerance_rad: float
) -> None:
    if candidate is None:
        return
    for index, known in enumerate(candidates):
        if _rotation_distance(candidate, known) <= tolerance_rad:
            candidates[index] = replace(
                known,
                provenance=tuple(sorted(set(known.provenance + candidate.provenance))),
                floor_face_ids=tuple(sorted(set(known.floor_face_ids + candidate.floor_face_ids))),
                wall_face_ids=tuple(sorted(set(known.wall_face_ids + candidate.wall_face_ids))),
            )
            return
    candidates.append(candidate)


def build_pose_catalog(
    mesh_path: str | Path,
    *,
    units: str = "mm",
    angular_tolerance_deg: float = 0.1,
    relative_distance_tolerance: float = 1e-6,
    rotation_tolerance_rad: float = 1e-6,
) -> PoseCatalog:
    """Enumerate all isolated theoretical face-edge floor-wall poses.

    The catalog is purely geometric and therefore independent of alpha/beta.
    Angle-dependent perturbation stability is applied in the next pipeline
    step.
    """

    mesh = load_solid_mesh(mesh_path, units=units)
    hull = mesh.convex_hull
    vertices = np.asarray(hull.vertices, dtype=float).copy()
    center_mass = np.asarray(mesh.center_mass, dtype=float).copy()
    vertices_centered = vertices - center_mass
    mesh_vertices_centered = np.asarray(mesh.vertices, dtype=float).copy() - center_mass
    mesh_edges = np.asarray(mesh.edges_unique, dtype=int).copy()
    mesh_faces = np.asarray(mesh.faces, dtype=int).copy()
    length_scale = float(np.max(hull.extents))
    distance_tolerance = max(length_scale * relative_distance_tolerance, 1e-9)

    support_faces = _extract_support_faces(
        hull,
        angular_tolerance_deg=angular_tolerance_deg,
        distance_tolerance_mm=distance_tolerance,
    )
    candidates: list[ContactPose] = []

    for face in support_faces:
        normal = np.asarray(face.normal_part, dtype=float)

        # Anchor this face on the floor, then rotate around floor normal Z
        # until every projected silhouette edge has been tested at the wall.
        floor_alignment = _rotation_from_to(normal, np.array([0.0, 0.0, -1.0]))
        floor_aligned = (floor_alignment @ vertices_centered.T).T
        for edge_start, edge_end in _silhouette_edges(
            floor_aligned, (0, 1), distance_tolerance
        ):
            edge = edge_end - edge_start
            theta = -math.atan2(float(edge[1]), float(edge[0]))
            for candidate_theta in (theta, theta + math.pi):
                rotation = _rotation_axis(2, candidate_theta) @ floor_alignment
                candidate = _make_candidate(
                    rotation,
                    vertices_centered,
                    mesh_vertices_centered,
                    mesh_edges,
                    mesh_faces,
                    support_faces,
                    distance_tolerance,
                    provenance=f"floor_face:{face.face_id}",
                )
                _add_candidate(candidates, candidate, rotation_tolerance_rad)

        # Mirror case: anchor the face on the wall, then rotate around wall
        # normal Y until a projected silhouette edge supports the floor.
        wall_alignment = _rotation_from_to(normal, np.array([0.0, -1.0, 0.0]))
        wall_aligned = (wall_alignment @ vertices_centered.T).T
        for edge_start, edge_end in _silhouette_edges(
            wall_aligned, (0, 2), distance_tolerance
        ):
            edge = edge_end - edge_start
            theta = math.atan2(float(edge[1]), float(edge[0]))
            for candidate_theta in (theta, theta + math.pi):
                rotation = _rotation_axis(1, candidate_theta) @ wall_alignment
                candidate = _make_candidate(
                    rotation,
                    vertices_centered,
                    mesh_vertices_centered,
                    mesh_edges,
                    mesh_faces,
                    support_faces,
                    distance_tolerance,
                    provenance=f"wall_face:{face.face_id}",
                )
                _add_candidate(candidates, candidate, rotation_tolerance_rad)

    candidates.sort(
        key=lambda pose: tuple(round(value, 12) for value in pose.quaternion_xyzw)
    )
    candidates = [replace(candidate, pose_id=index) for index, candidate in enumerate(candidates)]

    return PoseCatalog(
        source=str(Path(mesh_path).expanduser().resolve()),
        units=units,
        center_mass_part_mm=tuple(float(value) for value in center_mass),
        support_faces=support_faces,
        poses=tuple(candidates),
        contact_tolerance_mm=distance_tolerance,
        rotation_tolerance_rad=rotation_tolerance_rad,
    )
