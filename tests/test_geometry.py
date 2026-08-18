from pathlib import Path

import numpy as np

from chute_pose import inspect_mesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DF1A_STL = REPOSITORY_ROOT / "Werkstücke_STL_grob" / "Df1a.STL"


def test_df1a_is_a_valid_uniform_density_solid() -> None:
    report = inspect_mesh(DF1A_STL)

    assert report.units == "mm"
    assert report.watertight
    assert report.winding_consistent
    assert report.uniform_density_assumed
    np.testing.assert_allclose(report.extents_mm, [80.0, 69.28203583, 15.0], atol=1e-5)
    np.testing.assert_allclose(
        report.center_mass_mm,
        [40.00271810, 23.09872058, 5.37569569],
        atol=1e-5,
    )
    np.testing.assert_allclose(report.volume_mm3, 29174.232083, atol=1e-5)


def test_df1a_convex_hull_has_expected_complexity() -> None:
    report = inspect_mesh(DF1A_STL)

    assert report.hull_vertex_count == 11
    assert report.hull_face_count == 18
    assert report.hull_plane_count == 9

