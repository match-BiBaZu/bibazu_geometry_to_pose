from pathlib import Path
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chute_pose.classical import planar_reference, normalize_weights, solid_angle, _edge_lift, analyze_chute_classical, reseated_edge_lift
from chute_pose.contacts import build_pose_catalog

PARTS = Path(__file__).resolve().parents[1] / "Werkstücke_STL_grob"
SQUARE = np.array([[-1., -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])


def test_reference_cube_matches_analytic_solid_angle_and_critical_construction():
    result = planar_reference(SQUARE, np.array([0, 0, 1]), np.array([0, 0, 1]))
    assert result["omega_sr"] == pytest.approx(2 * math.pi / 3)
    expected_critical = 4 * math.atan(1 / (math.sqrt(2) * 2))
    assert result["critical_angles_sr"] == pytest.approx([expected_critical] * 4)
    assert result["crsa"] == pytest.approx(2 * math.pi / 3 - expected_critical)
    assert normalize_weights([result["crsa"]] * 6) == pytest.approx([1/6] * 6)
    assert normalize_weights([result["crsa"]] * 2, [2, 4]) == pytest.approx([1/3, 2/3])


def test_reference_translation_rotation_and_scale():
    com = np.array([0.3, 0.1, 1.4])
    normal = np.array([0., 0, 1])
    first = planar_reference(SQUARE, com, normal)
    rotation = Rotation.from_euler("xy", [45, 20], degrees=True).as_matrix()
    shift = np.array([10, 30, -7])
    second = planar_reference(SQUARE @ rotation.T + shift, rotation @ com + shift, rotation @ normal)
    scaled = planar_reference(SQUARE * 3, com * 3, normal)
    for key in ("standard_csa", "crsa"):
        assert second[key] == pytest.approx(first[key])
        assert scaled[key] == pytest.approx(first[key] / 3)


def test_displaced_center_changes_weights_and_critical_edges():
    centered = planar_reference(SQUARE, np.array([0, 0, 1]), np.array([0, 0, 1]))
    displaced = planar_reference(SQUARE, np.array([0.6, 0, 1]), np.array([0, 0, 1]))
    assert displaced["standard_csa"] != pytest.approx(centered["standard_csa"])
    assert len(set(round(v, 8) for v in displaced["critical_angles_sr"])) > 1


def test_normalization_rejects_invalid_data():
    for values in ([0, 0], [-1, 1], [float("nan")]):
        with pytest.raises(ValueError): normalize_weights(values)
    with pytest.raises(ValueError): normalize_weights([1, 2], [1, -1])


def test_fixed_pivot_gravity_saddle_and_wall_collision():
    vertices = np.array([[x, y, z] for x in (0, 2) for y in (2, 4) for z in (0, 2)], dtype=float)
    com = np.array([1., 3, 1])
    lift, reason = _edge_lift(vertices, com, np.array([0., 2, 0]), np.array([2., 2, 0]), np.array([0., 0, 1]), 1e-8)
    assert reason == "ok"
    assert lift == pytest.approx(math.sqrt(2) - 1)
    # Move the pivot to the wall: the same outward escape is now obstructed.
    shift = np.array([0, 2, 0])
    lift, reason = _edge_lift(vertices - shift, com - shift, np.array([0., 0, 0]), np.array([2., 0, 0]), np.array([0., 0, 1]), 1e-8)
    assert lift is None
    assert reason == "blocked_by_other_plane"


def test_chute_model_records_supported_and_unsupported_contacts():
    mesh = PARTS / "Df1a.STL"
    catalog = build_pose_catalog(mesh)
    results = analyze_chute_classical(mesh, catalog, [p.pose_id for p in catalog.poses], alpha_deg=45, beta_deg=0)
    assert any(v.standard_csa is not None for v in results.values())
    assert any(v.crsa is not None for v in results.values())
    for result in results.values():
        for score in (result.standard_csa, result.crsa):
            assert score is None or math.isfinite(score) and score >= 0
        for patch in result.patches:
            assert patch["normal_load_fraction"] == pytest.approx(0.5)


def test_reseated_horizontal_limit_matches_paper_lift_and_grid_refinement():
    vertices = np.array([[x, y, z] for x in (0, 2) for y in (2, 4) for z in (0, 2)], dtype=float)
    com = np.array([1., 3, 1])
    for step in (1.0, 0.5, 0.25):
        lift, reason = reseated_edge_lift(vertices, com, np.array([0., 2, 0]),
            np.array([2., 2, 0]), np.array([0., 0, 1]), np.array([1., 3, 0]),
            np.array([0., 0, 1]), step_deg=step)
        assert reason == "reseated_saddle"
        assert lift == pytest.approx(math.sqrt(2) - 1, abs=1e-8)
