from pathlib import Path

import pytest
from chute_pose.csa import _has_contact_reserve
from chute_pose.stability import StabilitySample

from chute_pose import (
    DEFAULT_CSA_ROBUST_THRESHOLD,
    analyze_contact_wrench_solid_angle,
    build_pose_catalog,
    filter_contact_wrench_solid_angle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"
KK1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Kk1a.STL"
DL4A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Dl4a.STL"


def test_cwsa_contact_reserve_does_not_require_forward_motion() -> None:
    stopped = StabilitySample(0.19, True, 0.14, -2.6, False, "would_not_move_in_positive_x")
    assert _has_contact_reserve(stopped, 1e-6)
    assert not _has_contact_reserve(
        StabilitySample(0.19, False, 0.14, 1.0, False, "no_force_moment_equilibrium"), 1e-6
    )
    assert not _has_contact_reserve(
        StabilitySample(0.19, True, 0.0, -2.6, False, "would_not_move_in_positive_x"), 1e-6
    )


def test_dl4a_cwsa_at_zero_longitudinal_slope_has_nonzero_distinct_scores() -> None:
    # Roadmap poses 9 and 22: both have contact reserve but fail the old
    # positive-X motion check at beta=0 for nonzero friction samples.
    analysis = analyze_contact_wrench_solid_angle(
        DL4A_STL, pose_ids=(15, 26), alpha_deg=45.0, beta_deg=0.0,
        mu_samples=2, direction_samples=8,
        friction_policy="range",
    )
    first, second = analysis.pose(15), analysis.pose(26)
    assert 0.0 < first.score <= 1.0
    assert 0.0 < second.score <= 1.0
    assert first.score != pytest.approx(second.score)
    assert first.nominal_contact_load_balance_index > 0.0
    assert second.nominal_contact_load_balance_index > 0.0
    assert analysis.to_dict()["requires_positive_x_acceleration"] is False


def test_cwsa_defaults_to_no_sliding_load_and_ignores_onset_settings() -> None:
    catalog = build_pose_catalog(DL4A_STL)
    static = analyze_contact_wrench_solid_angle(
        DL4A_STL, pose_ids=(15, 26), alpha_deg=45.0, beta_deg=0.0,
        direction_samples=8, catalog=catalog,
        onset_beta_deg=0.0, mu_samples=1,
    )
    sliding = analyze_contact_wrench_solid_angle(
        DL4A_STL, pose_ids=(15, 26), alpha_deg=45.0, beta_deg=0.0,
        direction_samples=8, catalog=catalog, friction_policy="range", mu_samples=2,
    )
    assert static.mu_values == (0.0,)
    assert static.gravity_chute_m_s2[0] == pytest.approx(0.0)
    assert static.to_dict()["load_model"] == "frictionless_support"
    assert sliding.to_dict()["load_model"] == "prescribed_sliding_friction"
    for pose in static.poses:
        assert pose.score >= sliding.pose(pose.pose_id).score - 1e-10


def test_cwsa_validates_explicit_friction_policy() -> None:
    with pytest.raises(ValueError, match="friction_policy"):
        analyze_contact_wrench_solid_angle(DF1A_STL, friction_policy="invalid")
    with pytest.raises(ValueError, match="mu_samples"):
        analyze_contact_wrench_solid_angle(DF1A_STL, friction_policy="range", mu_samples=1)


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
