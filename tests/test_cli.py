from chute_pose.cli import _build_parser, _format_pose_plot_metrics
from chute_pose.csa import (
    DEFAULT_CSA_CAP_HALF_ANGLE_DEG,
    DEFAULT_CSA_DIRECTION_SAMPLES,
    DEFAULT_CSA_ROBUST_THRESHOLD,
)
from chute_pose.roadmap import DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM


def test_roadmap_uses_tight_symmetry_tolerance_by_default() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["roadmap", "part.stl", "--output-dir", "roadmap", "--expected-symmetry", "C3"]
    )

    assert args.symmetry_tolerance_mm == DEFAULT_ROADMAP_SYMMETRY_TOLERANCE_MM
    assert args.expected_symmetry == "C3"


def test_rocking_remains_the_default_ranking_and_classifier() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["roadmap", "part.stl", "--output-dir", "roadmap"]
    )

    assert args.pose_ranking == "rocking"
    assert args.robustness_method == "rocking"
    assert args.minimum_csa_score == DEFAULT_CSA_ROBUST_THRESHOLD
    assert args.csa_cap_half_angle_deg == DEFAULT_CSA_CAP_HALF_ANGLE_DEG
    assert args.csa_direction_samples == DEFAULT_CSA_DIRECTION_SAMPLES
    assert args.minimum_braking_g == 0.0
    assert args.friction_policy == "zero"


def test_roadmap_accepts_the_legacy_face_face_braking_flag() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "roadmap",
            "part.stl",
            "--output-dir",
            "roadmap",
            "--minimum-face-face-braking-g",
            "0.25",
        ]
    )

    assert args.minimum_braking_g == 0.25


def test_csa_ranking_and_classification_can_be_selected_independently() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "roadmap",
            "part.stl",
            "--output-dir",
            "roadmap",
            "--pose-ranking",
            "csa",
            "--robustness-method",
            "csa",
            "--minimum-csa-score",
            "0.7",
            "--csa-cap-half-angle-deg",
            "7.5",
            "--csa-direction-samples",
            "96",
        ]
    )

    assert args.pose_ranking == "csa"
    assert args.robustness_method == "csa"
    assert args.minimum_csa_score == 0.7
    assert args.csa_cap_half_angle_deg == 7.5
    assert args.csa_direction_samples == 96


def test_exhaustive_friction_diagnostics_is_opt_in() -> None:
    parser = _build_parser()

    default_args = parser.parse_args(["stability", "part.stl"])
    exhaustive_args = parser.parse_args(
        ["stability", "part.stl", "--exhaustive-friction-diagnostics"]
    )

    assert not default_args.exhaustive_friction_diagnostics
    assert exhaustive_args.exhaustive_friction_diagnostics


def test_zero_friction_policy_is_the_generation_default() -> None:
    parser = _build_parser()

    default_args = parser.parse_args(["stability", "part.stl"])
    range_args = parser.parse_args(
        ["stability", "part.stl", "--friction-policy", "range"]
    )

    assert default_args.friction_policy == "zero"
    assert range_args.friction_policy == "range"

    roadmap_args = parser.parse_args(
        [
            "roadmap",
            "part.stl",
            "--output-dir",
            "output",
            "--friction-policy",
            "zero",
        ]
    )
    assert roadmap_args.friction_policy == "zero"
    disturbance_args = parser.parse_args(["disturbance", "part.stl"])
    assert disturbance_args.friction_policy == "zero"


def test_csa_plot_label_also_shows_the_rocking_barrier() -> None:
    label = _format_pose_plot_metrics(
        rocking_barrier_mm=1.23456,
        csa_stability_index=0.6789,
        show_csa=True,
    )

    assert "CWSA stability index: 0.679" in label
    assert "Rocking barrier: 1.235 mm" in label
