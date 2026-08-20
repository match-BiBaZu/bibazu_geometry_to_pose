from pathlib import Path

import pytest

pytest.importorskip("OCP")

from chute_pose import detect_rotational_symmetry, verify_step_symmetry


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARTS = REPOSITORY_ROOT / "Werkstücke_STL_grob"


def test_df1a_step_exactly_confirms_repaired_c3_symmetry() -> None:
    candidate = detect_rotational_symmetry(
        PARTS / "Df1a.STL", tolerance_mm=0.05
    )
    verification = verify_step_symmetry(PARTS / "Df1a.STEP", candidate)

    assert verification.status == "exact_confirmed"
    assert verification.exact_confirmed
    assert max(
        check.relative_symmetric_difference for check in verification.checks
    ) == pytest.approx(0.0, abs=1e-12)


def test_ql1i_step_exactly_confirms_c4_symmetry() -> None:
    candidate = detect_rotational_symmetry(PARTS / "Ql1i.STL")
    verification = verify_step_symmetry(PARTS / "Ql1i.STEP", candidate)

    assert verification.status == "exact_confirmed"
    assert verification.exact_confirmed
