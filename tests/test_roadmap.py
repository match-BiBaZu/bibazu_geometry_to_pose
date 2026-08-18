import base64
import json
from pathlib import Path

import pytest

from chute_pose import build_pose_roadmap
from chute_pose.roadmap import (
    PoseRoadmap,
    RoadmapEdge,
    RoadmapNode,
    find_best_route,
    geometric_reliability_score,
    render_pose_roadmap,
    roadmap_handover_dict,
    save_roadmap_json,
    save_roadmap_yaml,
    save_roadmap_yaml_readme,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
DL1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Dl1a.STL"
KK1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Kk1a.STL"
QL1I_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Ql1i.STL"


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
        main_face_ids=(0,),
        main_face_area_mm2=1.0,
        main_face_min_span_mm=1.0,
        opposite_x_min_height_mm=25.0,
        robust_barrier_threshold_mm=0.2,
        axis_tolerance_deg=1.0,
        nodes=(_node(1), _node(2), _node(3)),
        edges=edges,
        unresolved_metastable_node_ids=(),
    )


def test_df1a_roadmap_keeps_four_robust_and_seven_metastable_classes(
    tmp_path: Path,
) -> None:
    roadmap = build_pose_roadmap(DF1A_STL)

    assert len(roadmap.nodes) == 11
    assert sum(node.kind == "robust" for node in roadmap.nodes) == 4
    assert sum(node.kind == "metastable" for node in roadmap.nodes) == 7
    assert roadmap.main_face_id == 5
    assert roadmap.main_face_ids == (5,)
    assert roadmap.main_face_min_span_mm > 25.0
    assert not roadmap.unresolved_metastable_node_ids
    assert all(node.cad_status == "provisional" for node in roadmap.nodes)

    svg_path, png_path = render_pose_roadmap(roadmap, tmp_path / "Df1a_roadmap")
    assert svg_path.stat().st_size > 10_000
    assert png_path.stat().st_size > 10_000
    json_path = save_roadmap_json(roadmap, tmp_path / "Df1a_roadmap.json")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    thumbnail_data = json_payload["nodes"][0]["thumbnail_png_base64"]
    assert base64.b64decode(thumbnail_data, validate=True).startswith(b"\x89PNG")

    nodes = {node.node_id: node for node in roadmap.nodes}
    for edge in roadmap.edges:
        assert abs(edge.signed_angle_deg) <= 180.0 + 1e-8
        if edge.actuation in {"floor_main_neg_x", "floor_main_pos_x"}:
            assert nodes[edge.source].main_face_on_floor
            if edge.actuation == "floor_main_neg_x":
                assert edge.signed_angle_deg < 0.0
            else:
                assert edge.signed_angle_deg > 0.0
        elif edge.actuation in {"wall_main_neg_x", "wall_main_pos_x"}:
            assert nodes[edge.source].main_face_on_wall
            if edge.actuation == "wall_main_neg_x":
                assert edge.signed_angle_deg < 0.0
            else:
                assert edge.signed_angle_deg > 0.0
        elif edge.actuation in {"free_y", "free_z"}:
            assert edge.transition_kind == "actuated"
        elif edge.actuation == "passive":
            assert edge.transition_kind == "passive_tip"
            assert edge.escape_barrier_mm is not None
            assert edge.target in nodes

    actuations = {edge.actuation for edge in roadmap.edges}
    assert "floor_main_pos_x" in actuations
    assert "wall_main_neg_x" in actuations


def test_ql1i_main_face_is_too_narrow_for_opposite_x_actions() -> None:
    roadmap = build_pose_roadmap(QL1I_STL, geometry_status="verified")

    assert len(roadmap.main_face_ids) == 4
    assert roadmap.main_face_min_span_mm == pytest.approx(20.0)
    assert not {
        "floor_main_pos_x",
        "wall_main_neg_x",
    }.intersection(edge.actuation for edge in roadmap.edges)


def test_kk1a_continuous_symmetry_keeps_dominant_mantle_poses() -> None:
    roadmap = build_pose_roadmap(KK1A_STL)

    assert roadmap.symmetry_symbol == "Cinf"
    assert [node.node_id for node in roadmap.nodes] == [4, 5, 6, 8, 10, 11]
    assert [node.node_id for node in roadmap.nodes if node.kind == "robust"] == [
        5,
        10,
    ]
    assert [
        node.node_id for node in roadmap.nodes if node.kind == "metastable"
    ] == [4, 6, 8, 11]
    mantle_nodes = [
        node
        for node in roadmap.nodes
        if node.floor_contact_topology == node.wall_contact_topology == "edge"
    ]
    assert len(mantle_nodes) == 2
    assert all(node.rocking_barrier_mm > 4.8 for node in mantle_nodes)
    assert all(
        node.rocking_barrier_mm > 1.4
        for node in roadmap.nodes
        if node not in mantle_nodes
    )
    assert roadmap.main_face_min_span_mm < 25.0

    direct_robust_edges = {
        (edge.source, edge.target, edge.actuation, round(edge.signed_angle_deg, 3))
        for edge in roadmap.edges
        if edge.source in {5, 10}
        and edge.target in {5, 10}
        and not edge.settling_pose_ids
    }
    assert direct_robust_edges == {
        (5, 10, "free_y", 170.831),
        (5, 10, "free_z", -170.831),
        (10, 5, "free_y", -170.831),
        (10, 5, "free_z", 170.831),
    }

    relaxed_sources = {
        edge.source
        for edge in roadmap.edges
        if edge.target in {5, 10} and edge.settling_pose_ids
    }
    assert relaxed_sources == {4, 6, 8, 11}
    assert all(
        edge.settling_pose_ids in {(1,), (2,)}
        for edge in roadmap.edges
        if edge.settling_pose_ids
    )


def test_geometric_score_rewards_wide_deep_capture_basin() -> None:
    # Dl1a-like narrow/shallow end-face landing versus a tolerant rectangular
    # outlet landing. These are scores, deliberately not probabilities.
    narrow = geometric_reliability_score(5.0, 360.0, 0.05)[2]
    tolerant = geometric_reliability_score(35.0, 360.0, 0.25)[2]

    assert tolerant > narrow


def test_dl1a_free_axis_end_face_targets_score_below_observed_outlet_poses() -> None:
    roadmap = build_pose_roadmap(DL1A_STL, geometry_status="verified")
    incoming_scores: dict[int, list[float]] = {node.node_id: [] for node in roadmap.nodes}
    for edge in roadmap.edges:
        # Normal X actions between symmetry-equivalent main faces are a
        # separate actuator case. This regression compares the free Y/Z
        # capture basins which make the end-face landings difficult in use.
        if edge.actuation in {"free_y", "free_z"}:
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


def test_yaml_handover_contains_editable_experimental_transition_fields(
    tmp_path: Path,
) -> None:
    roadmap = _roadmap((_edge("direct", 1, 3, 0.5),))
    handover = roadmap_handover_dict(roadmap)

    assert handover["poses"][0]["planner_role"] == "stable_target"
    transition = handover["transitions"][0]
    assert transition["from_pose"] == 1
    assert transition["to_pose"] == 3
    assert transition["action"]["axis"] == "y"
    assert transition["action"]["axis_vector_chute"] == (0.0, 1.0, 0.0)
    assert transition["experimental"] == {
        "status": "untested",
        "trials": None,
        "successes": None,
        "empirical_success_rate": None,
        "difficulty_rating": None,
        "notes": "",
    }

    yaml_path = save_roadmap_yaml(roadmap, tmp_path / "roadmap.yaml")
    readme_path = save_roadmap_yaml_readme(tmp_path / "README.md")
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert yaml_text.startswith("---\n")
    assert 'format: "bibazu_pose_roadmap_handover"' in yaml_text
    assert "empirical_success_rate: null" in yaml_text
    assert "Erfolgswahrscheinlichkeit" in readme_path.read_text(encoding="utf-8")
