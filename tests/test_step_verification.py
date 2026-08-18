from pathlib import Path

import pytest

pytest.importorskip("OCP")

from chute_pose import detect_rotational_symmetry, verify_step_symmetry


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARTS = REPOSITORY_ROOT / "Werkstücke_STL_grob"


def test_df1a_step_reveals_practical_but_not_exact_c3_symmetry() -> None:
    candidate = detect_rotational_symmetry(
        PARTS / "Df1a.STL", tolerance_mm=0.5
    )
    verification = verify_step_symmetry(PARTS / "Df1a.STEP", candidate)

    assert verification.status == "practical_only_step_geometry_is_not_exact"
    assert not verification.exact_confirmed
    assert max(
        check.relative_symmetric_difference for check in verification.checks
    ) == pytest.approx(0.004038, abs=2e-5)


def test_ql1i_step_exactly_confirms_c4_symmetry() -> None:
    candidate = detect_rotational_symmetry(PARTS / "Ql1i.STL")
    verification = verify_step_symmetry(PARTS / "Ql1i.STEP", candidate)

    assert verification.status == "exact_confirmed"
    assert verification.exact_confirmed
