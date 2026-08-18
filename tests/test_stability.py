from pathlib import Path

import numpy as np

from chute_pose import analyze_pose_stability, estimate_equal_contact_friction


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"


def test_friction_estimate_uses_both_contacts() -> None:
    estimate = estimate_equal_contact_friction(
        onset_alpha_deg=45.0, onset_beta_deg=15.0
    )

    np.testing.assert_allclose(estimate.mu_static_estimate, 0.189468690981506)


def test_df1a_stability_baseline_at_operating_angles() -> None:
    analysis = analyze_pose_stability(DF1A_STL)

    assert len(analysis.poses) == 108
    assert len(analysis.stable_pose_ids) == 27
    assert len(analysis.friction_dependent_pose_ids) == 20
    assert len(analysis.rejected_pose_ids) == 61
    stable_counts: dict[tuple[str, str], int] = {}
    for result in analysis.poses:
        if result.stable_across_range:
            key = (result.floor_contact_type, result.wall_contact_type)
            stable_counts[key] = stable_counts.get(key, 0) + 1
            assert all(sample.acceleration_x_m_s2 >= 0.0 for sample in result.samples)
            assert result.minimum_pressure_margin > 0.0
    assert stable_counts == {
        ("edge", "face"): 8,
        ("face", "edge"): 7,
        ("face", "face"): 12,
    }


def test_all_confirmed_df1a_face_face_poses_pass() -> None:
    analysis = analyze_pose_stability(DF1A_STL)
    face_face = [
        result
        for result in analysis.poses
        if result.floor_contact_type == "face" and result.wall_contact_type == "face"
    ]

    assert len(face_face) == 12
    assert all(result.stable_across_range for result in face_face)
    assert min(result.minimum_pressure_margin for result in face_face) > 0.6
