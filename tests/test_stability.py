from pathlib import Path

import numpy as np

from chute_pose import (
    analyze_pose_stability,
    build_pose_catalog,
    estimate_equal_contact_friction,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
KF1I_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Kf1i.STL"


def test_friction_estimate_uses_both_contacts() -> None:
    estimate = estimate_equal_contact_friction(
        onset_alpha_deg=45.0, onset_beta_deg=15.0
    )

    np.testing.assert_allclose(estimate.mu_static_estimate, 0.189468690981506)


def test_df1a_stability_baseline_at_operating_angles() -> None:
    analysis = analyze_pose_stability(DF1A_STL, friction_policy="range")

    assert len(analysis.poses) == 90
    assert len(analysis.stable_pose_ids) == 24
    assert len(analysis.friction_dependent_pose_ids) == 21
    assert len(analysis.rejected_pose_ids) == 45
    stable_counts: dict[tuple[str, str], int] = {}
    for result in analysis.poses:
        if result.stable_across_range:
            key = (result.floor_contact_type, result.wall_contact_type)
            stable_counts[key] = stable_counts.get(key, 0) + 1
            assert all(sample.acceleration_x_m_s2 >= 0.0 for sample in result.samples)
            assert result.minimum_pressure_margin > 0.0
    assert stable_counts == {
        ("edge", "face"): 6,
        ("face", "edge"): 6,
        ("face", "face"): 12,
    }


def test_quasistatic_only_matches_exhaustive_stable_pose_results() -> None:
    catalog = build_pose_catalog(DF1A_STL)
    exhaustive = analyze_pose_stability(
        DF1A_STL,
        catalog=catalog,
        mu_samples=3,
        friction_policy="range",
    )
    accepted_only = analyze_pose_stability(
        DF1A_STL,
        catalog=catalog,
        mu_samples=3,
        quasistatic_only=True,
        friction_policy="range",
    )

    assert accepted_only.stable_pose_ids == exhaustive.stable_pose_ids
    exhaustive_by_id = {pose.pose_id: pose for pose in exhaustive.poses}
    for pose in accepted_only.poses:
        reference = exhaustive_by_id[pose.pose_id]
        assert pose.samples == reference.samples
        assert pose.minimum_pressure_margin == reference.minimum_pressure_margin

    serialized = accepted_only.to_dict()
    assert serialized["evaluation_mode"] == "quasistatic_accepted_only"
    assert serialized["input_pose_count"] == len(catalog.poses)
    assert serialized["friction_dependent_pose_ids"] is None
    assert serialized["rejected_pose_ids"] is None


def test_zero_friction_policy_is_default_and_retains_kf1i_equilibria() -> None:
    analysis = analyze_pose_stability(KF1I_STL)

    assert analysis.friction_policy == "zero"
    assert analysis.mu_values == (0.0,)
    assert analysis.stable_pose_ids == (0, 1, 2, 3, 4, 5)
    assert all(len(pose.samples) == 1 for pose in analysis.poses)
    serialized = analysis.to_dict()
    assert serialized["friction_policy"] == "zero"
    assert serialized["evaluation_mode"] == "zero_friction_nominal"
    assert serialized["friction_dependent_pose_ids"] is None
    assert serialized["rejected_pose_ids"] is None


def test_all_confirmed_df1a_face_face_poses_pass() -> None:
    analysis = analyze_pose_stability(DF1A_STL, friction_policy="range")
    face_face = [
        result
        for result in analysis.poses
        if result.floor_contact_type == "face" and result.wall_contact_type == "face"
    ]

    assert len(face_face) == 12
    assert all(result.stable_across_range for result in face_face)
    assert min(result.minimum_pressure_margin for result in face_face) > 0.6
    for result in face_face:
        assert result.minimum_contact_load_balance_index == (
            result.minimum_pressure_margin
        )
        assert result.to_dict()["minimum_contact_load_balance_index"] == (
            result.minimum_contact_load_balance_index
        )
