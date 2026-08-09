from pathlib import Path

import pytest

from chute_pose import build_pose_roadmap
from chute_pose.roadmap import (
    PoseRoadmap,
    RoadmapEdge,
    RoadmapNode,
    find_best_route,
    geometric_reliability_score,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
DL1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Dl1a.STL"


def _node(node_id: int, kind: str = "robust") -> RoadmapNode:
    return RoadmapNode(
        node_id=node_id,
        pose_ids=(node_id,),
        kind=kind,  # type: ignore[arg-type]
        cad_status="verified",
        representative_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        floor_contact_topology="face",
        wall_contact_topology="face",
        rocking_barrier_mm=1.0 if kind == "robust" else 0.05,
        main_face_on_floor=False,
        main_face_on_wall=False,
    )


def _edge(edge_id: str, source: int, target: int, score: float) -> RoadmapEdge:
    return RoadmapEdge(
        edge_id=edge_id,
        source=source,
        target=target,
        transition_kind="actuated",
        actuation="free_y",
        axis_chute=(0.0, 1.0, 0.0),
        signed_angle_deg=45.0,
        capture_interval_deg=(35.0, 55.0),
        capture_width_deg=20.0,
        capture_fraction=score,
        target_barrier_score=1.0,
        geometric_score=score,
    )


def _roadmap(edges: tuple[RoadmapEdge, ...]) -> PoseRoadmap:
    return PoseRoadmap(
        schema_version=1,
        source="synthetic.stl",
        geometry_status="verified",
        alpha_deg=45.0,
        beta_deg=20.0,
        symmetry_symbol="C1",
        symmetry_tolerance_mm=0.0,
        main_face_id=0,
        main_face_area_mm2=1.0,
        robust_barrier_threshold_mm=0.2,
        axis_tolerance_deg=1.0,
        nodes=(_node(1), _node(2), _node(3)),
        edges=edges,
        unresolved_metastable_node_ids=(),
    )


def test_df1a_roadmap_keeps_four_robust_and_seven_metastable_classes() -> None:
    roadmap = build_pose_roadmap(DF1A_STL)

    assert len(roadmap.nodes) == 11
    assert sum(node.kind == "robust" for node in roadmap.nodes) == 4
    assert sum(node.kind == "metastable" for node in roadmap.nodes) == 7
    assert roadmap.main_face_id == 5
    assert not roadmap.unresolved_metastable_node_ids
    assert all(node.cad_status == "provisional" for node in roadmap.nodes)

    nodes = {node.node_id: node for node in roadmap.nodes}
    for edge in roadmap.edges:
        assert abs(edge.signed_angle_deg) <= 180.0 + 1e-8
        if edge.actuation == "floor_main_neg_x":
            assert edge.signed_angle_deg < 0.0
            assert nodes[edge.source].main_face_on_floor
        elif edge.actuation == "wall_main_pos_x":
            assert edge.signed_angle_deg > 0.0
            assert nodes[edge.source].main_face_on_wall
        elif edge.actuation in {"free_y", "free_z"}:
            assert edge.transition_kind == "actuated"
        elif edge.actuation == "passive":
            assert edge.transition_kind == "passive_tip"
            assert edge.escape_barrier_mm is not None
            assert edge.target in nodes


def test_geometric_score_rewards_wide_deep_capture_basin() -> None:
    # Dl1a-like narrow/shallow end-face landing versus a tolerant rectangular
    # outlet landing. These are scores, deliberately not probabilities.
    narrow = geometric_reliability_score(5.0, 360.0, 0.05)[2]
    tolerant = geometric_reliability_score(35.0, 360.0, 0.25)[2]

    assert tolerant > narrow


def test_dl1a_end_face_targets_score_below_observed_rectangular_outlet_poses() -> None:
    roadmap = build_pose_roadmap(DL1A_STL, geometry_status="verified")
    incoming_scores: dict[int, list[float]] = {node.node_id: [] for node in roadmap.nodes}
    for edge in roadmap.edges:
        if edge.transition_kind == "actuated":
            incoming_scores[edge.target].append(edge.geometric_score)

    observed_outlet_nodes = (15, 16, 31, 34)
    end_face_nodes = (32, 91, 112, 137)
    best_outlet_scores = [max(incoming_scores[node]) for node in observed_outlet_nodes]
    best_end_face_scores = [max(incoming_scores[node]) for node in end_face_nodes]

    assert min(best_outlet_scores) > max(best_end_face_scores)
    assert all(roadmap.node(node).kind == "robust" for node in observed_outlet_nodes)
    assert all(roadmap.node(node).kind == "metastable" for node in end_face_nodes)
    assert all(node.cad_status == "verified" for node in roadmap.nodes)


def test_route_prefers_more_reliable_path_and_caps_actuations() -> None:
    roadmap = _roadmap(
        (
            _edge("direct", 1, 3, 0.5),
            _edge("via-a", 1, 2, 0.9),
            _edge("via-b", 2, 3, 0.9),
        )
    )

    route = find_best_route(roadmap, 1, 3, max_actions=4)
    assert route.node_path == (1, 2, 3)
    assert route.actuation_count == 2
    assert route.geometric_score == pytest.approx(0.81)

    direct_only = find_best_route(roadmap, 1, 3, max_actions=1)
    assert direct_only.edge_ids == ("direct",)
    with pytest.raises(ValueError, match="between 0 and 4"):
        find_best_route(roadmap, 1, 3, max_actions=5)
