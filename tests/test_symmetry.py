from pathlib import Path

from chute_pose import build_symmetry_reduced_catalog, detect_rotational_symmetry


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARTS = REPOSITORY_ROOT / "Werkstücke_STL_grob"


def test_df1a_practical_threefold_symmetry_and_known_pose_classes() -> None:
    catalog, reduced = build_symmetry_reduced_catalog(
        PARTS / "Df1a.STL", tolerance_mm=0.5
    )

    assert len(catalog.poses) == 108
    assert reduced.symmetry.symbol == "C3"
    assert reduced.symmetry.order == 3
    assert max(
        element.mapping_error_mm for element in reduced.symmetry.elements
    ) < 0.32
    assert reduced.class_for_pose(9).pose_ids == (9, 12, 32)
    assert reduced.class_for_pose(24).pose_ids == (24, 26, 28)
    assert reduced.class_for_pose(60).pose_ids == (60, 61, 86)
    assert reduced.class_for_pose(35).pose_ids == (35, 105, 106)


def test_ql1i_fourfold_symmetry_is_detected_automatically() -> None:
    symmetry = detect_rotational_symmetry(PARTS / "Ql1i.STL")

    assert symmetry.symbol == "C4"
    assert symmetry.order == 4
