from pathlib import Path

import pytest

from chute_pose import (
    DEFAULT_CSA_ROBUST_THRESHOLD,
    analyze_contact_wrench_solid_angle,
    build_pose_catalog,
    filter_contact_wrench_solid_angle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
KK1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Kk1a.STL"


def test_cwsa_is_absolute_and_separates_reference_df1a_poses() -> None:
    catalog = build_pose_catalog(DF1A_STL)
    together = analyze_contact_wrench_solid_angle(
        DF1A_STL,
        pose_ids=(23, 45),
        mu_samples=2,
        direction_samples=8,
        catalog=catalog,
    )
    alone = analyze_contact_wrench_solid_angle(
        DF1A_STL,
        pose_ids=(23,),
        mu_samples=2,
        direction_samples=8,
        catalog=catalog,
    )

    robust = together.pose(23)
    metastable = together.pose(45)
    assert robust.score == pytest.approx(alone.pose(23).score, abs=1e-10)
    assert 0.0 <= metastable.score < DEFAULT_CSA_ROBUST_THRESHOLD
    assert DEFAULT_CSA_ROBUST_THRESHOLD < robust.score <= 1.0
    assert robust.feasible_solid_angle_sr <= robust.cap_solid_angle_sr
    assert robust.floor_patch_solid_angle_sr >= 0.0
    assert robust.wall_patch_solid_angle_sr >= 0.0


def test_cwsa_filter_validates_and_reports_cutoff() -> None:
    analysis = analyze_contact_wrench_solid_angle(
        DF1A_STL,
        pose_ids=(23, 45),
        mu_samples=2,
        direction_samples=8,
    )
    filtered = filter_contact_wrench_solid_angle(
        analysis, minimum_score=DEFAULT_CSA_ROBUST_THRESHOLD
    )

    assert filtered.accepted_pose_ids == (23,)
    assert filtered.rejected_pose_ids == (45,)
    assert not filtered.not_applicable_pose_ids
    with pytest.raises(ValueError, match="between 0 and 1"):
        filter_contact_wrench_solid_angle(analysis, minimum_score=1.01)


def test_cwsa_marks_continuous_rolling_contacts_not_applicable() -> None:
    catalog = build_pose_catalog(KK1A_STL)
    rolling_pose = next(
        pose
        for pose in catalog.poses
        if pose.floor_contact_dimension == pose.wall_contact_dimension == 1
    )
    analysis = analyze_contact_wrench_solid_angle(
        KK1A_STL,
        pose_ids=(rolling_pose.pose_id,),
        mu_samples=2,
        direction_samples=8,
        catalog=catalog,
    )

    result = analysis.pose(rolling_pose.pose_id)
    assert not result.applicable
    assert result.reason == "continuous_rolling_contact_switching_not_modelled"
    assert filter_contact_wrench_solid_angle(
        analysis
    ).not_applicable_pose_ids == (rolling_pose.pose_id,)
