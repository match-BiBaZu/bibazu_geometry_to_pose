"""Pose-class roadmap, actuator adjacency and open-loop route planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation

from .contacts import ContactPose, PoseCatalog, build_pose_catalog
from .disturbance import analyze_disturbance_robustness
from .equivalence import PracticalPoseClass, cluster_practical_contact_poses
from .frame import ChuteFrame
from .geometry import load_solid_mesh
from .rocking import (
    analyze_rocking_barriers,
    filter_finite_disturbance_robustness,
)
from .stability import analyze_pose_stability
from .symmetry import detect_rotational_symmetry


NodeKind = Literal["robust", "metastable"]
TransitionKind = Literal["actuated", "passive_tip"]
ActuationKind = Literal[
    "floor_main_neg_x", "wall_main_pos_x", "free_y", "free_z", "passive"
]


@dataclass(frozen=True, slots=True)
class RoadmapNode:
    node_id: int
    pose_ids: tuple[int, ...]
    kind: NodeKind
    cad_status: str
    representative_quaternion_xyzw: tuple[float, float, float, float]
    floor_contact_topology: str
    wall_contact_topology: str
    rocking_barrier_mm: float
    main_face_on_floor: bool
    main_face_on_wall: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoadmapNode":
        return cls(
            node_id=int(value["node_id"]),
            pose_ids=tuple(int(item) for item in value["pose_ids"]),
            kind=value["kind"],
            cad_status=str(value.get("cad_status", "provisional")),
            representative_quaternion_xyzw=tuple(
                float(item) for item in value["representative_quaternion_xyzw"]
            ),  # type: ignore[arg-type]
            floor_contact_topology=str(value["floor_contact_topology"]),
            wall_contact_topology=str(value["wall_contact_topology"]),
            rocking_barrier_mm=float(value["rocking_barrier_mm"]),
            main_face_on_floor=bool(value["main_face_on_floor"]),
            main_face_on_wall=bool(value["main_face_on_wall"]),
        )


@dataclass(frozen=True, slots=True)
class RoadmapEdge:
    edge_id: str
    source: int
    target: int
    transition_kind: TransitionKind
    actuation: ActuationKind
    axis_chute: tuple[float, float, float]
    signed_angle_deg: float
    capture_interval_deg: tuple[float, float] | None
    capture_width_deg: float
    capture_fraction: float
    target_barrier_score: float
    geometric_score: float
    escape_barrier_mm: float | None = None
    saddle_angle_deg: float | None = None
    axis_error_deg: float = 0.0

    @property
    def actuation_count(self) -> int:
        return 1 if self.transition_kind == "actuated" else 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actuation_count"] = self.actuation_count
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoadmapEdge":
        interval = value.get("capture_interval_deg")
        return cls(
            edge_id=str(value["edge_id"]),
            source=int(value["source"]),
            target=int(value["target"]),
            transition_kind=value["transition_kind"],
            actuation=value["actuation"],
            axis_chute=tuple(float(item) for item in value["axis_chute"]),  # type: ignore[arg-type]
            signed_angle_deg=float(value["signed_angle_deg"]),
            capture_interval_deg=(
                tuple(float(item) for item in interval) if interval is not None else None
            ),  # type: ignore[arg-type]
            capture_width_deg=float(value["capture_width_deg"]),
            capture_fraction=float(value["capture_fraction"]),
            target_barrier_score=float(value["target_barrier_score"]),
            geometric_score=float(value["geometric_score"]),
            escape_barrier_mm=(
                float(value["escape_barrier_mm"])
                if value.get("escape_barrier_mm") is not None
                else None
            ),
            saddle_angle_deg=(
                float(value["saddle_angle_deg"])
                if value.get("saddle_angle_deg") is not None
                else None
            ),
            axis_error_deg=float(value.get("axis_error_deg", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class PoseRoadmap:
    schema_version: int
    source: str
    geometry_status: str
    alpha_deg: float
    beta_deg: float
    symmetry_symbol: str
    symmetry_tolerance_mm: float
    main_face_id: int
    main_face_area_mm2: float
    robust_barrier_threshold_mm: float
    axis_tolerance_deg: float
    nodes: tuple[RoadmapNode, ...]
    edges: tuple[RoadmapEdge, ...]
    unresolved_metastable_node_ids: tuple[int, ...]

    def node(self, node_or_pose_id: int) -> RoadmapNode:
        for node in self.nodes:
            if node.node_id == node_or_pose_id or node_or_pose_id in node.pose_ids:
                return node
        raise KeyError(node_or_pose_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "geometry_status": self.geometry_status,
            "alpha_deg": self.alpha_deg,
            "beta_deg": self.beta_deg,
            "symmetry_symbol": self.symmetry_symbol,
            "symmetry_tolerance_mm": self.symmetry_tolerance_mm,
            "main_face_id": self.main_face_id,
            "main_face_area_mm2": self.main_face_area_mm2,
            "robust_barrier_threshold_mm": self.robust_barrier_threshold_mm,
            "axis_tolerance_deg": self.axis_tolerance_deg,
            "node_counts": {
                "total": len(self.nodes),
                "robust": sum(node.kind == "robust" for node in self.nodes),
                "metastable": sum(node.kind == "metastable" for node in self.nodes),
            },
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "unresolved_metastable_node_ids": self.unresolved_metastable_node_ids,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseRoadmap":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported roadmap schema version.")
        return cls(
            schema_version=1,
            source=str(value["source"]),
            geometry_status=str(value["geometry_status"]),
            alpha_deg=float(value["alpha_deg"]),
            beta_deg=float(value["beta_deg"]),
            symmetry_symbol=str(value["symmetry_symbol"]),
            symmetry_tolerance_mm=float(value["symmetry_tolerance_mm"]),
            main_face_id=int(value["main_face_id"]),
            main_face_area_mm2=float(value["main_face_area_mm2"]),
            robust_barrier_threshold_mm=float(value["robust_barrier_threshold_mm"]),
            axis_tolerance_deg=float(value["axis_tolerance_deg"]),
            nodes=tuple(RoadmapNode.from_dict(item) for item in value["nodes"]),
            edges=tuple(RoadmapEdge.from_dict(item) for item in value["edges"]),
            unresolved_metastable_node_ids=tuple(
                int(item) for item in value.get("unresolved_metastable_node_ids", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class RoutePlan:
    start_node: int
    target_node: int
    node_path: tuple[int, ...]
    edge_ids: tuple[str, ...]
    actuation_count: int
    total_abs_angle_deg: float
    geometric_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ACTION_SPECS: dict[str, tuple[np.ndarray, tuple[float, float]]] = {
    "floor_main_neg_x": (np.array([1.0, 0.0, 0.0]), (-180.0, 0.0)),
    "wall_main_pos_x": (np.array([1.0, 0.0, 0.0]), (0.0, 180.0)),
    "free_y": (np.array([0.0, 1.0, 0.0]), (-180.0, 180.0)),
    "free_z": (np.array([0.0, 0.0, 1.0]), (-180.0, 180.0)),
}


def geometric_reliability_score(
    capture_width_deg: float,
    available_angle_span_deg: float,
    target_barrier_mm: float,
    robust_barrier_threshold_mm: float = 0.20,
) -> tuple[float, float, float]:
    """Return capture fraction, barrier score and their transparent product."""

    if available_angle_span_deg <= 0.0 or robust_barrier_threshold_mm <= 0.0:
        raise ValueError("Reliability reference spans must be positive.")
    capture = float(np.clip(capture_width_deg / available_angle_span_deg, 0.0, 1.0))
    barrier = float(np.clip(target_barrier_mm / robust_barrier_threshold_mm, 0.0, 1.0))
    return capture, barrier, capture * barrier


def _seated_height_mm(
    rotation: np.ndarray, vertices_centered: np.ndarray, gravity: np.ndarray
) -> float:
    points = (rotation @ vertices_centered.T).T
    potential_per_mass = (
        float(gravity[1]) * float(np.min(points[:, 1]))
        + float(gravity[2]) * float(np.min(points[:, 2]))
    )
    return potential_per_mass / float(np.linalg.norm(gravity))


def _rotation_about(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, dtype=float) * math.radians(angle_deg)).as_matrix()


def _basin_half_width_deg(
    target_rotation: np.ndarray,
    axis: np.ndarray,
    direction: float,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    *,
    maximum_deg: float = 90.0,
    coarse_step_deg: float = 1.0,
    refine_step_deg: float = 0.1,
) -> float:
    """Locate the first 1-D seated-energy crest away from a target pose."""

    baseline = _seated_height_mm(target_rotation, vertices_centered, gravity)
    previous = baseline
    peak = baseline
    coarse_boundary = maximum_deg
    found_descent = False
    angle = coarse_step_deg
    while angle <= maximum_deg + 1e-9:
        candidate = _rotation_about(axis, direction * angle) @ target_rotation
        height = _seated_height_mm(candidate, vertices_centered, gravity)
        peak = max(peak, height)
        if peak > baseline + 1e-7 and height < previous - 1e-6:
            coarse_boundary = max(0.0, angle - coarse_step_deg)
            found_descent = True
            break
        previous = height
        angle += coarse_step_deg
    if not found_descent:
        return maximum_deg

    low = max(0.0, coarse_boundary - coarse_step_deg)
    high = min(maximum_deg, coarse_boundary + coarse_step_deg)
    samples = np.arange(low, high + 0.5 * refine_step_deg, refine_step_deg)
    heights = [
        _seated_height_mm(
            _rotation_about(axis, direction * float(value)) @ target_rotation,
            vertices_centered,
            gravity,
        )
        for value in samples
    ]
    return float(samples[int(np.argmax(heights))])


def _class_main_face_placement(
    pose_class: PracticalPoseClass,
    poses: dict[int, ContactPose],
    main_face_id: int,
) -> tuple[bool, bool]:
    return (
        all(main_face_id in poses[pose_id].floor_face_ids for pose_id in pose_class.pose_ids),
        all(main_face_id in poses[pose_id].wall_face_ids for pose_id in pose_class.pose_ids),
    )


def _best_actuated_relation(
    source_class: PracticalPoseClass,
    target_class: PracticalPoseClass,
    poses: dict[int, ContactPose],
    action: str,
    main_face_id: int,
    axis_tolerance_deg: float,
) -> tuple[float, float, int, int] | None:
    axis, domain = _ACTION_SPECS[action]
    best: tuple[float, float, int, int] | None = None
    for source_pose_id in source_class.pose_ids:
        source_pose = poses[source_pose_id]
        if action == "floor_main_neg_x" and main_face_id not in source_pose.floor_face_ids:
            continue
        if action == "wall_main_pos_x" and main_face_id not in source_pose.wall_face_ids:
            continue
        source_rotation = np.asarray(source_pose.rotation_chute_from_part, dtype=float)
        for target_pose_id in target_class.pose_ids:
            target_rotation = np.asarray(
                poses[target_pose_id].rotation_chute_from_part, dtype=float
            )
            rotvec = Rotation.from_matrix(target_rotation @ source_rotation.T).as_rotvec()
            angle_rad = float(np.linalg.norm(rotvec))
            if angle_rad <= 1e-9:
                continue
            unit = rotvec / angle_rad
            alignment = float(np.dot(unit, axis))
            axis_error = math.degrees(math.acos(float(np.clip(abs(alignment), -1.0, 1.0))))
            if axis_error > axis_tolerance_deg:
                continue
            signed_angle = math.degrees(angle_rad) * (1.0 if alignment >= 0.0 else -1.0)
            if abs(abs(signed_angle) - 180.0) <= 1e-7:
                if action == "floor_main_neg_x":
                    signed_angle = -180.0
                elif action == "wall_main_pos_x":
                    signed_angle = 180.0
            if signed_angle < domain[0] - 1e-7 or signed_angle > domain[1] + 1e-7:
                continue
            candidate = (axis_error, signed_angle, source_pose_id, target_pose_id)
            if best is None or (candidate[0], abs(candidate[1])) < (
                best[0],
                abs(best[1]),
            ):
                best = candidate
    return best


def _actuated_edges(
    classes: tuple[PracticalPoseClass, ...],
    nodes_by_id: dict[int, RoadmapNode],
    poses: dict[int, ContactPose],
    main_face_id: int,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    robust_barrier_threshold_mm: float,
    axis_tolerance_deg: float,
) -> list[RoadmapEdge]:
    edges: list[RoadmapEdge] = []
    class_node_ids = {
        pose_class.class_id: pose_class.representative_pose_id for pose_class in classes
    }
    for source_class in classes:
        source_id = class_node_ids[source_class.class_id]
        for target_class in classes:
            if source_class.class_id == target_class.class_id:
                continue
            target_id = class_node_ids[target_class.class_id]
            for action, (axis, domain) in _ACTION_SPECS.items():
                if (
                    action == "floor_main_neg_x"
                    and not nodes_by_id[source_id].main_face_on_floor
                ):
                    continue
                if (
                    action == "wall_main_pos_x"
                    and not nodes_by_id[source_id].main_face_on_wall
                ):
                    continue
                relation = _best_actuated_relation(
                    source_class,
                    target_class,
                    poses,
                    action,
                    main_face_id,
                    axis_tolerance_deg,
                )
                if relation is None:
                    continue
                axis_error, signed_angle, _, target_pose_id = relation
                target_rotation = np.asarray(
                    poses[target_pose_id].rotation_chute_from_part, dtype=float
                )
                left_width = _basin_half_width_deg(
                    target_rotation, axis, -1.0, vertices_centered, gravity
                )
                right_width = _basin_half_width_deg(
                    target_rotation, axis, 1.0, vertices_centered, gravity
                )
                interval = (
                    max(domain[0], signed_angle - left_width),
                    min(domain[1], signed_angle + right_width),
                )
                capture_width = max(0.0, interval[1] - interval[0])
                target_node = nodes_by_id[target_id]
                capture, barrier_score, score = geometric_reliability_score(
                    capture_width,
                    domain[1] - domain[0],
                    target_node.rocking_barrier_mm,
                    robust_barrier_threshold_mm,
                )
                edges.append(
                    RoadmapEdge(
                        edge_id=f"a{len(edges)}:{source_id}->{target_id}:{action}",
                        source=source_id,
                        target=target_id,
                        transition_kind="actuated",
                        actuation=action,  # type: ignore[arg-type]
                        axis_chute=tuple(float(value) for value in axis),
                        signed_angle_deg=float(signed_angle),
                        capture_interval_deg=tuple(float(value) for value in interval),
                        capture_width_deg=capture_width,
                        capture_fraction=capture,
                        target_barrier_score=barrier_score,
                        geometric_score=max(score, 1e-12),
                        axis_error_deg=float(axis_error),
                    )
                )
    return edges


def _geodesic_escape(
    source_rotation: np.ndarray,
    target_rotation: np.ndarray,
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    *,
    step_deg: float = 1.0,
) -> tuple[float, float, float, tuple[float, float, float]] | None:
    relative = target_rotation @ source_rotation.T
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    angle_rad = float(np.linalg.norm(rotvec))
    if angle_rad <= 1e-9:
        return None
    axis = rotvec / angle_rad
    total_angle_deg = math.degrees(angle_rad)
    sample_count = max(2, int(math.ceil(total_angle_deg / step_deg)) + 1)
    sample_angles = np.linspace(0.0, total_angle_deg, sample_count)
    heights = np.asarray(
        [
            _seated_height_mm(
                _rotation_about(axis, float(angle)) @ source_rotation,
                vertices_centered,
                gravity,
            )
            for angle in sample_angles
        ]
    )
    peak_index = int(np.argmax(heights))
    barrier = max(0.0, float(heights[peak_index] - heights[0]))
    endpoint_change = float(heights[-1] - heights[0])
    return (
        barrier,
        endpoint_change,
        float(sample_angles[peak_index]),
        tuple(float(value) for value in axis),
    )


def _passive_edges(
    classes: tuple[PracticalPoseClass, ...],
    nodes_by_id: dict[int, RoadmapNode],
    poses: dict[int, ContactPose],
    vertices_centered: np.ndarray,
    gravity: np.ndarray,
    robust_barrier_threshold_mm: float,
) -> tuple[list[RoadmapEdge], tuple[int, ...]]:
    edges: list[RoadmapEdge] = []
    unresolved: list[int] = []
    for source_class in classes:
        source_id = source_class.representative_pose_id
        if nodes_by_id[source_id].kind != "metastable":
            continue
        candidates: list[
            tuple[float, float, float, tuple[float, float, float], int]
        ] = []
        for target_class in classes:
            target_id = target_class.representative_pose_id
            if target_id == source_id:
                continue
            best = None
            for source_pose_id in source_class.pose_ids:
                source_rotation = np.asarray(
                    poses[source_pose_id].rotation_chute_from_part, dtype=float
                )
                for target_pose_id in target_class.pose_ids:
                    target_rotation = np.asarray(
                        poses[target_pose_id].rotation_chute_from_part, dtype=float
                    )
                    escape = _geodesic_escape(
                        source_rotation,
                        target_rotation,
                        vertices_centered,
                        gravity,
                    )
                    if escape is None:
                        continue
                    if best is None or (escape[0], abs(escape[1])) < (
                        best[0],
                        abs(best[1]),
                    ):
                        best = escape
            if best is not None and best[0] < robust_barrier_threshold_mm and best[1] <= 1e-4:
                candidates.append((*best, target_id))
        if not candidates:
            unresolved.append(source_id)
            continue
        minimum_barrier = min(value[0] for value in candidates)
        selected = [
            value for value in candidates if value[0] <= minimum_barrier + 0.02
        ]
        for barrier, _, saddle_angle, axis, target_id in selected:
            target_node = nodes_by_id[target_id]
            escape_score = float(
                np.clip(1.0 - barrier / robust_barrier_threshold_mm, 0.0, 1.0)
            )
            barrier_score = float(
                np.clip(
                    target_node.rocking_barrier_mm / robust_barrier_threshold_mm,
                    0.0,
                    1.0,
                )
            )
            score = max(1e-12, escape_score * barrier_score)
            edges.append(
                RoadmapEdge(
                    edge_id=f"p{len(edges)}:{source_id}->{target_id}",
                    source=source_id,
                    target=target_id,
                    transition_kind="passive_tip",
                    actuation="passive",
                    axis_chute=axis,
                    signed_angle_deg=0.0,
                    capture_interval_deg=None,
                    capture_width_deg=0.0,
                    capture_fraction=escape_score,
                    target_barrier_score=barrier_score,
                    geometric_score=score,
                    escape_barrier_mm=float(barrier),
                    saddle_angle_deg=float(saddle_angle),
                )
            )
    return edges, tuple(unresolved)


def build_pose_roadmap(
    mesh_path: str | Path,
    *,
    alpha_deg: float = 45.0,
    beta_deg: float = 20.0,
    onset_alpha_deg: float = 45.0,
    onset_beta_deg: float = 15.0,
    symmetry_tolerance_mm: float = 0.5,
    angular_tolerance_deg: float = 1.0,
    surface_displacement_tolerance_mm: float = 0.5,
    robust_barrier_threshold_mm: float = 0.20,
    minimum_face_face_braking_g: float = 0.10,
    geometry_status: str = "provisional",
) -> PoseRoadmap:
    """Build the robust/metastable physical pose roadmap for one part."""

    if geometry_status not in {"provisional", "verified"}:
        raise ValueError("geometry_status must be 'provisional' or 'verified'.")
    catalog = build_pose_catalog(mesh_path)
    nominal = analyze_pose_stability(
        mesh_path,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        onset_alpha_deg=onset_alpha_deg,
        onset_beta_deg=onset_beta_deg,
        catalog=catalog,
    )
    nominal_ids = nominal.stable_pose_ids
    disturbance = analyze_disturbance_robustness(
        mesh_path,
        pose_ids=nominal_ids,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        onset_alpha_deg=onset_alpha_deg,
        onset_beta_deg=onset_beta_deg,
        catalog=catalog,
    )
    rocking = analyze_rocking_barriers(
        mesh_path,
        pose_ids=nominal_ids,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        catalog=catalog,
    )
    robust_filter = filter_finite_disturbance_robustness(
        rocking,
        disturbance,
        catalog,
        minimum_barrier_height_mm=robust_barrier_threshold_mm,
        minimum_face_face_braking_g=minimum_face_face_braking_g,
    )
    robust_pose_ids = set(robust_filter.accepted_pose_ids)
    mesh = load_solid_mesh(mesh_path)
    vertices_centered_full = np.asarray(mesh.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    hull_vertices_centered = np.asarray(mesh.convex_hull.vertices, dtype=float) - np.asarray(
        mesh.center_mass, dtype=float
    )
    symmetry = detect_rotational_symmetry(
        mesh_path, tolerance_mm=symmetry_tolerance_mm
    )
    clustering = cluster_practical_contact_poses(
        catalog,
        vertices_centered_full,
        nominal_ids,
        symmetry=symmetry,
        angular_tolerance_deg=angular_tolerance_deg,
        surface_displacement_tolerance_mm=max(
            surface_displacement_tolerance_mm, symmetry.tolerance_mm
        ),
    )
    main_face = max(catalog.support_faces, key=lambda value: value.area_mm2)
    poses = {pose.pose_id: pose for pose in catalog.poses}
    barriers = {value.pose_id: value for value in rocking.barriers}
    nodes: list[RoadmapNode] = []
    for pose_class in clustering.classes:
        representative = poses[pose_class.representative_pose_id]
        main_floor, main_wall = _class_main_face_placement(
            pose_class, poses, main_face.face_id
        )
        nodes.append(
            RoadmapNode(
                node_id=pose_class.representative_pose_id,
                pose_ids=pose_class.pose_ids,
                kind=(
                    "robust"
                    if all(pose_id in robust_pose_ids for pose_id in pose_class.pose_ids)
                    else "metastable"
                ),
                cad_status=geometry_status,
                representative_quaternion_xyzw=representative.quaternion_xyzw,
                floor_contact_topology=representative.floor_contact_topology,
                wall_contact_topology=representative.wall_contact_topology,
                rocking_barrier_mm=min(
                    barriers[pose_id].barrier_height_mm for pose_id in pose_class.pose_ids
                ),
                main_face_on_floor=main_floor,
                main_face_on_wall=main_wall,
            )
        )
    nodes_by_id = {node.node_id: node for node in nodes}
    gravity = ChuteFrame(alpha_deg=alpha_deg, beta_deg=beta_deg).gravity_chute()
    actuated = _actuated_edges(
        clustering.classes,
        nodes_by_id,
        poses,
        main_face.face_id,
        hull_vertices_centered,
        gravity,
        robust_barrier_threshold_mm,
        angular_tolerance_deg,
    )
    passive, unresolved = _passive_edges(
        clustering.classes,
        nodes_by_id,
        poses,
        hull_vertices_centered,
        gravity,
        robust_barrier_threshold_mm,
    )
    return PoseRoadmap(
        schema_version=1,
        source=str(Path(mesh_path).expanduser().resolve()),
        geometry_status=geometry_status,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        symmetry_symbol=symmetry.symbol,
        symmetry_tolerance_mm=symmetry.tolerance_mm,
        main_face_id=main_face.face_id,
        main_face_area_mm2=main_face.area_mm2,
        robust_barrier_threshold_mm=robust_barrier_threshold_mm,
        axis_tolerance_deg=angular_tolerance_deg,
        nodes=tuple(nodes),
        edges=tuple(actuated + passive),
        unresolved_metastable_node_ids=unresolved,
    )


def find_best_route(
    roadmap: PoseRoadmap,
    start_pose_id: int,
    target_pose_id: int,
    *,
    max_actions: int = 4,
) -> RoutePlan:
    """Maximise geometric route score with at most four open-loop impulses."""

    if max_actions < 0 or max_actions > 4:
        raise ValueError("max_actions must be between 0 and 4.")
    start = roadmap.node(start_pose_id).node_id
    target = roadmap.node(target_pose_id).node_id
    if start == target:
        return RoutePlan(start, target, (start,), (), 0, 0.0, 1.0)
    outgoing: dict[int, list[RoadmapEdge]] = {}
    for edge in roadmap.edges:
        outgoing.setdefault(edge.source, []).append(edge)

    queue: list[tuple[float, int, float, int, tuple[int, ...], tuple[str, ...]]] = [
        (0.0, 0, 0.0, start, (start,), ())
    ]
    best: dict[tuple[int, int], tuple[float, float]] = {(start, 0): (0.0, 0.0)}
    while queue:
        cost, action_count, angle_sum, node_id, nodes, edge_ids = heapq.heappop(queue)
        if node_id == target:
            return RoutePlan(
                start_node=start,
                target_node=target,
                node_path=nodes,
                edge_ids=edge_ids,
                actuation_count=action_count,
                total_abs_angle_deg=angle_sum,
                geometric_score=math.exp(-cost),
            )
        for edge in outgoing.get(node_id, ()):
            new_actions = action_count + edge.actuation_count
            if new_actions > max_actions:
                continue
            if edge.transition_kind == "passive_tip" and edge.target in nodes:
                continue
            edge_score = float(np.clip(edge.geometric_score, 1e-12, 1.0))
            new_cost = cost - math.log(edge_score)
            new_angle = angle_sum + (
                abs(edge.signed_angle_deg) if edge.transition_kind == "actuated" else 0.0
            )
            key = (edge.target, new_actions)
            previous = best.get(key)
            if previous is not None and (new_cost, new_angle) >= previous:
                continue
            best[key] = (new_cost, new_angle)
            heapq.heappush(
                queue,
                (
                    new_cost,
                    new_actions,
                    new_angle,
                    edge.target,
                    nodes + (edge.target,),
                    edge_ids + (edge.edge_id,),
                ),
            )
    raise ValueError(
        f"No route from pose {start_pose_id} to {target_pose_id} within {max_actions} actuations."
    )


def save_roadmap_json(roadmap: PoseRoadmap, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(roadmap.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_roadmap_json(path: str | Path) -> PoseRoadmap:
    source = Path(path).expanduser().resolve()
    return PoseRoadmap.from_dict(json.loads(source.read_text(encoding="utf-8")))


def save_roadmap_graphml(roadmap: PoseRoadmap, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for node in roadmap.nodes:
        graph.add_node(
            node.node_id,
            pose_ids="/".join(str(value) for value in node.pose_ids),
            kind=node.kind,
            cad_status=node.cad_status,
            rocking_barrier_mm=node.rocking_barrier_mm,
            floor_contact=node.floor_contact_topology,
            wall_contact=node.wall_contact_topology,
            main_face_on_floor=node.main_face_on_floor,
            main_face_on_wall=node.main_face_on_wall,
        )
    for edge in roadmap.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            transition_kind=edge.transition_kind,
            actuation=edge.actuation,
            signed_angle_deg=edge.signed_angle_deg,
            capture_width_deg=edge.capture_width_deg,
            geometric_score=edge.geometric_score,
            escape_barrier_mm=(
                edge.escape_barrier_mm if edge.escape_barrier_mm is not None else -1.0
            ),
        )
    nx.write_graphml(graph, destination)
    return destination


def render_pose_roadmap(
    roadmap: PoseRoadmap,
    output_stem: str | Path,
) -> tuple[Path, Path]:
    """Render robust nodes prominently and metastable nodes as quiet waypoints."""

    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node.node_id for node in roadmap.nodes)
    graph.add_edges_from((edge.source, edge.target, {"edge": edge}) for edge in roadmap.edges)
    positions = nx.spring_layout(graph, seed=42, k=1.25)
    figure, axis = plt.subplots(figsize=(14, 9), facecolor="white")
    axis.set_axis_off()
    robust = [node.node_id for node in roadmap.nodes if node.kind == "robust"]
    metastable = [node.node_id for node in roadmap.nodes if node.kind == "metastable"]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=robust,
        node_size=1650,
        node_color="#2b8cbe",
        edgecolors="#084081",
        linewidths=2.2,
        ax=axis,
    )
    meta_collection = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=metastable,
        node_size=850,
        node_color="#e5e7eb",
        edgecolors="#6b7280",
        linewidths=1.5,
        ax=axis,
    )
    meta_collection.set_linestyle("--")
    labels = {
        node.node_id: "/".join(str(value) for value in node.pose_ids)
        for node in roadmap.nodes
    }
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8, ax=axis)
    colors = {
        "floor_main_neg_x": "#1f77b4",
        "wall_main_pos_x": "#d62728",
        "free_y": "#2ca02c",
        "free_z": "#9467bd",
        "passive": "#7f7f7f",
    }
    for edge in roadmap.edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(edge.source, edge.target)],
            edge_color=colors[edge.actuation],
            style="dashed" if edge.transition_kind == "passive_tip" else "solid",
            width=1.1 if edge.transition_kind == "passive_tip" else 1.8,
            alpha=0.72,
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.08",
            ax=axis,
        )
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for edge in roadmap.edges:
        if edge.transition_kind == "passive_tip":
            label = f"passiv Δh={edge.escape_barrier_mm:.3f}mm"
        else:
            label = (
                f"{edge.actuation} {edge.signed_angle_deg:+.1f}° "
                f"w={edge.capture_width_deg:.1f}° s={edge.geometric_score:.3f}"
            )
        edge_labels.setdefault((edge.source, edge.target), []).append(label)
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels={key: "\n".join(value) for key, value in edge_labels.items()},
        font_size=6,
        rotate=False,
        ax=axis,
    )
    title = (
        f"{Path(roadmap.source).stem}: Posenroadmap — "
        f"{sum(node.kind == 'robust' for node in roadmap.nodes)} robust, "
        f"{sum(node.kind == 'metastable' for node in roadmap.nodes)} metastabil"
    )
    axis.set_title(title, fontsize=16, pad=18)
    legend_items = [
        Line2D([0], [0], color=colors["floor_main_neg_x"], lw=2, label="−X, Hauptflaeche Boden"),
        Line2D([0], [0], color=colors["wall_main_pos_x"], lw=2, label="+X, Hauptflaeche Wand"),
        Line2D([0], [0], color=colors["free_y"], lw=2, label="freie Y-Rotation"),
        Line2D([0], [0], color=colors["free_z"], lw=2, label="freie Z-Rotation"),
        Line2D([0], [0], color=colors["passive"], lw=1.5, ls="--", label="passives Kippen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2b8cbe", markeredgecolor="#084081", markersize=11, label="robust"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e5e7eb", markeredgecolor="#6b7280", markersize=8, label="metastabil"),
    ]
    axis.legend(handles=legend_items, loc="upper left", fontsize=8, frameon=True)
    if roadmap.geometry_status == "provisional":
        figure.text(
            0.5,
            0.02,
            "VORLAEUFIG — bekannt fehlerhaftes CAD; konkrete Uebergaenge neu berechnen",
            ha="center",
            color="#b91c1c",
            fontsize=11,
            weight="bold",
        )
    figure.tight_layout(rect=(0.01, 0.04, 0.99, 0.98))
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def export_pose_roadmap(
    roadmap: PoseRoadmap, output_dir: str | Path
) -> tuple[Path, Path, Path, Path]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(roadmap.source).stem
    json_path = save_roadmap_json(roadmap, destination / f"{stem}_roadmap.json")
    graphml_path = save_roadmap_graphml(roadmap, destination / f"{stem}_roadmap.graphml")
    svg_path, png_path = render_pose_roadmap(
        roadmap, destination / f"{stem}_roadmap"
    )
    return json_path, graphml_path, svg_path, png_path
