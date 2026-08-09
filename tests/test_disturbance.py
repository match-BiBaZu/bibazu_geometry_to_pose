from pathlib import Path

from chute_pose import (
    analyze_disturbance_robustness,
    build_pose_catalog,
    filter_disturbance_robustness,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARTS = REPOSITORY_ROOT / "Werkstücke_STL_grob"


def test_df1a_face_face_pose_has_finite_positive_disturbance_reserve() -> None:
    catalog = build_pose_catalog(PARTS / "Df1a.STL")
    analysis = analyze_disturbance_robustness(
        PARTS / "Df1a.STL", pose_ids=[9], mu_samples=3, catalog=catalog
    )
    capacity = analysis.capacities[0]

    assert capacity.critical_braking_g > 0.5
    assert capacity.critical_torque_normalized > 0.2


def test_dl1a_long_axis_torque_separates_observed_mirrored_classes() -> None:
    catalog = build_pose_catalog(PARTS / "Dl1a.STL")
    analysis = analyze_disturbance_robustness(
        PARTS / "Dl1a.STL",
        pose_ids=[15, 32, 48],
        mu_samples=3,
        catalog=catalog,
    )
    capacities = {value.pose_id: value for value in analysis.capacities}

    # Pose 15 represents an observed longitudinal class. Pose 48 is the
    # visually plausible mirror class that rolls immediately about its long
    # principal axis. Pose 32 is a tall face-face class sensitive to braking.
    assert (
        capacities[15].critical_torque_normalized
        > 5.0 * capacities[48].critical_torque_normalized
    )
    assert capacities[15].critical_braking_g > 1.0
    assert capacities[32].critical_braking_g < 0.02


def test_provisional_thresholds_reproduce_dl1a_observation() -> None:
    catalog = build_pose_catalog(PARTS / "Dl1a.STL")
    representatives = [15, 16, 31, 32, 34, 48, 49, 51, 52, 91, 112, 119, 128, 137]
    analysis = analyze_disturbance_robustness(
        PARTS / "Dl1a.STL",
        pose_ids=representatives,
        mu_samples=3,
        catalog=catalog,
    )
    result = filter_disturbance_robustness(analysis)

    assert result.accepted_pose_ids == (15, 16, 31, 34)
