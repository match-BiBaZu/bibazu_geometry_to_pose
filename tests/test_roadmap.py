import base64
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.image import imread

from chute_pose import build_pose_roadmap
from chute_pose.roadmap import (
    _ROADMAP_AXIS_COLORS,
    PoseRoadmap,
    RoadmapEdge,
    RoadmapNode,
    _format_roadmap_edge_plot_label,
    _format_roadmap_node_plot_label,
    _roadmap_edge_curvatures,
    _roadmap_edge_label_position,
    _roadmap_plot_positions,
    _roadmap_total_path_length,
    _roadmap_transition_axis,
    _separate_roadmap_angle_labels,
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
DF4A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df4a.STL"
RL4I_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Rl4i.STL"


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


def test_df1a_roadmap_keeps_four_robust_and_four_metastable_classes(
    tmp_path: Path,
) -> None:
    roadmap = build_pose_roadmap(
        DF1A_STL,
        symmetry_tolerance_mm=0.05,
        friction_policy="range",
    )

    assert len(roadmap.nodes) == 8
    assert sum(node.kind == "robust" for node in roadmap.nodes) == 4
    assert sum(node.kind == "metastable" for node in roadmap.nodes) == 4
    assert roadmap.symmetry_symbol == "C3"
    assert roadmap.symmetry_tolerance_mm == pytest.approx(0.05)
    assert roadmap.friction_policy == "range"
    assert roadmap.main_face_id == 4
    assert roadmap.main_face_ids == (4,)
    assert roadmap.main_face_min_span_mm > 25.0
    assert not roadmap.unresolved_metastable_node_ids
    assert all(node.cad_status == "provisional" for node in roadmap.nodes)
    assert [node.node_id for node in roadmap.nodes] == list(range(8))
    assert [node.node_id for node in roadmap.nodes if node.kind == "robust"] == [
        0,
        1,
        2,
        3,
    ]
    assert [node.node_id for node in roadmap.nodes if node.kind == "metastable"] == [
        4,
        5,
        6,
        7,
    ]
    assert [node.pose_ids for node in roadmap.nodes] == [
        (23, 25, 27),
        (9, 12, 31),
        (34, 87, 88),
        (50, 51, 74),
        (45, 66, 81),
        (38, 75, 85),
        (37, 76, 86),
        (46, 67, 82),
    ]
    assert [node.rocking_barrier_mm for node in roadmap.nodes] == sorted(
        (node.rocking_barrier_mm for node in roadmap.nodes), reverse=True
    )
    svg_path, png_path = render_pose_roadmap(roadmap, tmp_path / "Df1a_roadmap")
    assert svg_path.stat().st_size > 10_000
    assert png_path.stat().st_size > 10_000
    rendered_pixels = imread(png_path)
    assert rendered_pixels.shape[1] / rendered_pixels.shape[0] == pytest.approx(1.5)
    stable_svg_path, stable_png_path = render_pose_roadmap(
        roadmap,
        tmp_path / "Df1a_roadmap_stable",
        stable_only=True,
    )
    assert stable_svg_path.stat().st_size > 5_000
    assert stable_png_path.stat().st_size > 5_000
    assert "stable pose roadmap" in stable_svg_path.read_text(encoding="utf-8")
    json_path = save_roadmap_json(roadmap, tmp_path / "Df1a_roadmap.json")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["nodes"][0]["node_id"] == 0
    assert json_payload["nodes"][0]["original_catalog_pose_id"] == 23
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


def test_csa_roadmap_node_label_also_shows_rocking_barrier() -> None:
    node = replace(_node(4), csa_stability_index=0.71234)

    label = _format_roadmap_node_plot_label(node, show_csa=True)

    assert label == "Rocking barrier 1.000 mm\nCWSA 0.712"


def test_roadmap_angle_labels_are_separated() -> None:
    figure, axis = plt.subplots()
    annotations = [
        axis.annotate(
            f"+{angle:.1f}°",
            (0.5, 0.5),
            xytext=(0.0, 0.0),
            textcoords="offset points",
            bbox={"facecolor": "white", "edgecolor": "none"},
        )
        for angle in range(0, 90, 10)
    ]

    _separate_roadmap_angle_labels(figure, annotations)
    renderer = figure.canvas.get_renderer()
    bounds = [
        annotation.get_bbox_patch().get_window_extent(renderer)
        for annotation in annotations
        if annotation.get_bbox_patch() is not None
    ]
    plt.close(figure)

    assert all(
        not left.overlaps(right)
        for index, left in enumerate(bounds)
        for right in bounds[index + 1 :]
    )


def test_roadmap_plot_uses_axis_colours_for_every_transition_kind() -> None:
    x_edge = replace(
        _edge("x", 1, 2, 0.8),
        actuation="floor_main_neg_x",
        axis_chute=(1.0, 0.0, 0.0),
    )
    y_edge = _edge("y", 1, 2, 0.8)
    z_edge = replace(
        _edge("z", 1, 2, 0.8),
        actuation="free_z",
        axis_chute=(0.0, 0.0, 1.0),
    )
    passive_z_edge = replace(
        _edge("passive-z", 2, 1, 0.8),
        transition_kind="passive_tip",
        actuation="passive",
        axis_chute=(0.1, -0.2, 0.9),
        escape_barrier_mm=0.1,
    )

    assert _roadmap_transition_axis(x_edge) == "x"
    assert _roadmap_transition_axis(y_edge) == "y"
    assert _roadmap_transition_axis(z_edge) == "z"
    assert _roadmap_transition_axis(passive_z_edge) == "z"
    assert _ROADMAP_AXIS_COLORS == {
        "x": "#d62728",
        "y": "#2ca02c",
        "z": "#1f77b4",
    }
    assert _format_roadmap_edge_plot_label(y_edge) == "+45.0°"
    assert _format_roadmap_edge_plot_label(
        replace(x_edge, signed_angle_deg=-90.0)
    ) == "−90.0°"
    assert _format_roadmap_edge_plot_label(
        replace(passive_z_edge, signed_angle_deg=-12.5)
    ) == "−12.5°"


def test_roadmap_plot_layout_uses_compact_common_aspect_grid() -> None:
    nodes = (_node(0),) + tuple(
        _node(node_id, "metastable") for node_id in range(1, 19)
    )
    edges = tuple(
        _edge(f"edge-{node_id}", 0, node_id, 0.5) for node_id in range(1, 19)
    )
    roadmap = replace(_roadmap(()), nodes=nodes, edges=edges)

    initial_positions, _, _ = _roadmap_plot_positions(
        roadmap, minimize_path_length=False
    )
    positions, columns, rows = _roadmap_plot_positions(roadmap)
    repeated_positions, repeated_columns, repeated_rows = _roadmap_plot_positions(
        roadmap
    )

    y_values = sorted({float(position[1]) for position in positions.values()}, reverse=True)
    assert columns == 6
    assert rows == 4
    assert all(
        sum(position[1] == y_value for position in positions.values()) <= columns
        for y_value in y_values
    )
    assert np.allclose(np.diff(y_values), -3.0)
    assert len({tuple(position) for position in positions.values()}) == len(nodes)
    assert _roadmap_total_path_length(
        roadmap, positions
    ) <= _roadmap_total_path_length(roadmap, initial_positions)
    assert repeated_columns == columns
    assert repeated_rows == rows
    assert all(
        np.array_equal(positions[node_id], repeated_positions[node_id])
        for node_id in positions
    )


def test_parallel_roadmap_edges_receive_distinct_curves() -> None:
    roadmap = _roadmap(
        (
            _edge("first", 1, 2, 0.5),
            _edge("second", 1, 2, 0.6),
            _edge("reverse", 2, 1, 0.7),
            _edge("long", 1, 3, 0.8),
        )
    )
    positions = {
        1: np.array((-1.2, 0.0)),
        2: np.array((1.2, -2.6)),
        3: np.array((-1.2, -10.4)),
    }

    curvatures = _roadmap_edge_curvatures(roadmap, positions)

    assert curvatures["first"] != curvatures["second"]
    assert abs(curvatures["long"]) > abs(curvatures["reverse"])
    assert not np.array_equal(
        _roadmap_edge_label_position(roadmap.edges[0], positions, curvatures["first"]),
        _roadmap_edge_label_position(roadmap.edges[1], positions, curvatures["second"]),
    )


def test_df1a_can_use_cwsa_for_ranking_and_robustness() -> None:
    roadmap = build_pose_roadmap(
        DF1A_STL,
        symmetry_tolerance_mm=0.05,
        pose_ranking_method="csa",
        robustness_method="csa",
        csa_direction_samples=8,
        friction_policy="range",
    )

    assert roadmap.pose_ranking_method == "csa"
    assert roadmap.robustness_method == "csa"
    assert roadmap.minimum_csa_score == pytest.approx(0.65)
    assert not roadmap.csa_rocking_fallback_pose_ids
    assert sum(node.kind == "robust" for node in roadmap.nodes) == 4
    assert sum(node.kind == "metastable" for node in roadmap.nodes) == 4
    assert all(node.csa_applicable for node in roadmap.nodes)
    assert all(node.csa_stability_index is not None for node in roadmap.nodes)
    scores = [
        node.csa_stability_index
        for node in roadmap.nodes
        if node.csa_stability_index is not None
    ]
    assert [round(value, 6) for value in scores] == sorted(
        (round(value, 6) for value in scores), reverse=True
    )
    assert all(
        node.csa_stability_index >= roadmap.minimum_csa_score
        for node in roadmap.nodes
        if node.kind == "robust" and node.csa_stability_index is not None
    )
    assert all(
        node.csa_stability_index < roadmap.minimum_csa_score
        for node in roadmap.nodes
        if node.kind == "metastable" and node.csa_stability_index is not None
    )
    handover = roadmap_handover_dict(roadmap)
    assert handover["classification"]["pose_ranking_method"] == "csa"
    assert handover["classification"]["robustness_method"] == "csa"
    assert handover["poses"][0]["csa_stability_index"] == pytest.approx(
        roadmap.nodes[0].csa_stability_index
    )
    loaded = PoseRoadmap.from_dict(roadmap.to_dict())
    assert loaded.pose_ranking_method == "csa"
    assert loaded.robustness_method == "csa"
    assert loaded.nodes[0].csa_stability_index == pytest.approx(
        roadmap.nodes[0].csa_stability_index
    )


def test_df1a_roadmap_uses_zero_friction_input_poses_by_default() -> None:
    roadmap = build_pose_roadmap(
        DF1A_STL,
        symmetry_tolerance_mm=0.05,
    )

    assert roadmap.friction_policy == "zero"
    assert len(roadmap.nodes) == 15
    assert roadmap.to_dict()["friction_policy"] == "zero"
    assert roadmap_handover_dict(roadmap)["classification"]["friction_policy"] == (
        "zero"
    )
    assert PoseRoadmap.from_dict(roadmap.to_dict()).friction_policy == "zero"


def test_schema_one_roadmap_without_csa_fields_still_loads() -> None:
    payload = _roadmap(()).to_dict()
    for key in (
        "pose_ranking_method",
        "robustness_method",
        "minimum_csa_score",
        "csa_cap_half_angle_deg",
        "csa_direction_samples",
        "csa_rocking_fallback_pose_ids",
        "friction_policy",
    ):
        payload.pop(key)
    for node in payload["nodes"]:
        for key in tuple(key for key in node if key.startswith("csa_")):
            node.pop(key)

    loaded = PoseRoadmap.from_dict(payload)

    assert loaded.pose_ranking_method == "rocking"
    assert loaded.robustness_method == "rocking"
    assert all(node.csa_stability_index is None for node in loaded.nodes)


def test_ql1i_main_face_is_too_narrow_for_opposite_x_actions() -> None:
    roadmap = build_pose_roadmap(QL1I_STL, geometry_status="verified")

    assert len(roadmap.main_face_ids) == 4
    assert roadmap.main_face_min_span_mm == pytest.approx(20.0)
    assert roadmap.opposite_x_min_height_mm == 25.0
    assert not {
        "floor_main_pos_x",
        "wall_main_neg_x",
    }.intersection(edge.actuation for edge in roadmap.edges)


def test_rl4i_blocks_narrow_opposite_x_support_and_rejects_weak_edge_poses() -> None:
    roadmap = build_pose_roadmap(RL4I_STL, geometry_status="verified")

    assert roadmap.main_face_min_span_mm <= roadmap.opposite_x_min_height_mm
    assert not {
        "floor_main_pos_x",
        "wall_main_neg_x",
    }.intersection(edge.actuation for edge in roadmap.edges)
    assert all(roadmap.node(node_id).kind == "metastable" for node_id in (12, 13, 14, 15))


def test_df4a_uses_disturbance_reserves_in_addition_to_rocking_barrier() -> None:
    roadmap = build_pose_roadmap(DF4A_STL, geometry_status="verified")

    assert roadmap.node(10).kind == "robust"
    assert all(roadmap.node(node_id).kind == "metastable" for node_id in (11, 12, 13))


def test_kk1a_continuous_symmetry_keeps_dominant_mantle_poses() -> None:
    roadmap = build_pose_roadmap(KK1A_STL)

    assert roadmap.symmetry_symbol == "Cinf"
    assert [node.node_id for node in roadmap.nodes] == [0, 1, 2, 3, 4, 5]
    assert [node.rocking_barrier_mm for node in roadmap.nodes] == sorted(
        (node.rocking_barrier_mm for node in roadmap.nodes), reverse=True
    )
    node_by_original_id = {
        node.original_catalog_pose_id: node for node in roadmap.nodes
    }
    robust_node_ids = {
        node.node_id for node in roadmap.nodes if node.kind == "robust"
    }
    metastable_node_ids = {
        node.node_id for node in roadmap.nodes if node.kind == "metastable"
    }
    assert robust_node_ids == {
        node_by_original_id[1].node_id,
        node_by_original_id[9].node_id,
    }
    assert metastable_node_ids == {
        node_by_original_id[value].node_id for value in (0, 3, 6, 11)
    }
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
        if edge.source in robust_node_ids
        and edge.target in robust_node_ids
        and not edge.settling_pose_ids
    }
    pose_1 = node_by_original_id[1].node_id
    pose_9 = node_by_original_id[9].node_id
    assert direct_robust_edges == {
        (pose_1, pose_9, "free_y", 170.823),
        (pose_1, pose_9, "free_z", -170.823),
        (pose_9, pose_1, "free_y", -170.823),
        (pose_9, pose_1, "free_z", 170.823),
    }

    relaxed_sources = {
        edge.source
        for edge in roadmap.edges
        if edge.target in robust_node_ids and edge.settling_pose_ids
    }
    assert relaxed_sources == metastable_node_ids
    assert all(
        edge.settling_pose_ids in {(2,), (5,), (8,), (10,)}
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
    roadmap = build_pose_roadmap(
        DL1A_STL,
        geometry_status="verified",
        friction_policy="range",
    )
    incoming_scores: dict[int, list[float]] = {node.node_id: [] for node in roadmap.nodes}
    for edge in roadmap.edges:
        # Normal X actions between symmetry-equivalent main faces are a
        # separate actuator case. This regression compares the free Y/Z
        # capture basins which make the end-face landings difficult in use.
        if edge.actuation in {"free_y", "free_z"}:
            incoming_scores[edge.target].append(edge.geometric_score)

    observed_outlet_nodes = tuple(
        roadmap.node(original_id).node_id for original_id in (15, 16, 31, 34)
    )
    end_face_nodes = tuple(
        roadmap.node(original_id).node_id for original_id in (32, 91, 112, 137)
    )
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
    assert handover["poses"][0]["original_catalog_pose_id"] == 1
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
